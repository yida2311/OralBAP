import os
import argparse
import time
import cv2
import math
import joblib
from numpy.core.numeric import full
import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader 
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import OralDataset, OralSlide, OralDatasetSim, Transformer, TransformerVal, TransformerSim, inverseTransformerSim
from models import Unet, UnetTA
# from utils.loss import CrossEntropyLoss, SegTALoss, SegMTLoss
from loss.loss_segMT import SegTALoss, SegMTLoss
from utils.lr_scheduler import LR_Scheduler
from utils.metric import ConfusionMatrix, AverageMeter
from utils.state_dict import model_Single2Parallel, save_ckpt_model, model_load_state_dict
from utils.vis_util import class_to_RGB, RGB_mapping_to_class
from utils.util import seed_everything, argParser, update_writer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
SEED = 552
seed_everything(SEED)

distributed = False
# if torch.cuda.device_count() > 1:
#     distributed = True
if distributed:
    # DPP 1
    dist.init_process_group('nccl')
    # DPP 2
    local_rank = dist.get_rank()
    print(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
else:
    device = torch.device("cuda:0")
    local_rank = 0
    
def main(cfg, device, local_rank=0):
    print(cfg.task_name)
    if local_rank == 0:
        if not os.path.exists(cfg.model_path): 
            os.makedirs(cfg.model_path)
        if not os.path.exists(cfg.log_path):
            os.makedirs(cfg.log_path)
        if not os.path.exists(cfg.writer_path):
            os.makedirs(cfg.writer_path)
        # if not os.path.exists(cfg.sim_output_path): 
        #     os.makedirs(cfg.sim_output_path)
        if not os.path.exists(cfg.coarse_output_path): 
            os.makedirs(cfg.coarse_output_path)
        if not os.path.exists(cfg.fine_output_path): 
            os.makedirs(cfg.fine_output_path)

    ### MODEL INIT
    if cfg.model == 'unet':
        model = Unet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
    elif cfg.model == 'unetTA':
        model = UnetTA(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.modelTA_cfg)

    if distributed:
        model = model_Single2Parallel(model, device, local_rank)
    else:
        model.to(device)
    ### DATASET
    dataset_train = OralDataset(
        cfg.trainset_cfg["img_dir"],
        cfg.trainset_cfg["mask_dir"],
        cfg.trainset_cfg["meta_file"], 
        label=cfg.trainset_cfg["label"], 
        transform=Transformer(crop_size=cfg.crop_size),
    )
    if distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        dataloader_train = DataLoader(dataset_train, 
                                    num_workers=cfg.num_workers,
                                    batch_size=cfg.batch_size,
                                    sampler=sampler_train)
    else:
        dataloader_train = DataLoader(dataset_train, 
                                    num_workers=cfg.num_workers,
                                    batch_size=cfg.batch_size,
                                    shuffle=True)
    
    dataset_coarse = OralSlide(
        cfg.coarseset_cfg["slide_list"],
        cfg.coarseset_cfg["img_dir"],
        cfg.coarseset_cfg["mask_dir"],
        cfg.coarseset_cfg["meta_file"],
        slide_mask_dir=cfg.coarseset_cfg["slide_mask_dir"],
        label=cfg.coarseset_cfg["label"], 
        transform=TransformerVal()
    )
    dataset_fine = OralSlide(
        cfg.fineset_cfg["slide_list"],
        cfg.fineset_cfg["img_dir"],
        cfg.fineset_cfg["mask_dir"],
        cfg.fineset_cfg["meta_file"],
        slide_mask_dir=cfg.fineset_cfg["slide_mask_dir"],
        label=cfg.fineset_cfg["label"], 
        transform=TransformerVal()
    )
    ### LOSS
    if cfg.loss == 'seg_ta':
        criterion = SegTALoss(**cfg.loss_cfg[cfg.loss])
    elif cfg.loss == 'seg_mt':
        criterion = SegMTLoss(**cfg.loss_cfg[cfg.loss])
    elif cfg.loss == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
    criterion = criterion.cuda()
    ### SOLVER
    acc_step = cfg.acc_step   # for gradient accumulation
    num_epochs = cfg.num_epochs
    learning_rate = cfg.lr
    evaluation = cfg.evaluation
    val_vis = cfg.val_vis
    batch_time = AverageMeter("BatchTime", ':3.3f')
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = LR_Scheduler(cfg.scheduler, learning_rate, num_epochs, len(dataloader_train))
    metrics = ConfusionMatrix(cfg.n_class)
    best_pred_fine = 0.0
    best_epoch = 0
    print("start training......")
    ### LOG INIT
    if local_rank == 0:
        f_log = open(cfg.log_path + cfg.task_name + ".log", 'w')
        log = cfg.task_name + '\n'
        for k, v in cfg.__dict__.items():
            log += str(k) + ' = ' + str(v) + '\n'
        print(log+'\n')
        f_log.write(log)
        f_log.flush()
    ### WRITER INIT
    if local_rank == 0:
        writer = SummaryWriter(log_dir=cfg.writer_path)
    writer_info = {}

    ### Running
    evaluator = SlideInference(cfg.n_class, cfg.num_workers, cfg.batch_size, cfg)
    # evaluator = SlideInference(cfg.n_class, cfg.num_workers, 8)
    #===========================================================
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        num_batch = len(dataloader_train)
        tbar = tqdm(dataloader_train)
        train_loss = 0

        ### Training
        start_time = time.time()
        model.train()
        for i_batch, sample in enumerate(tbar):
            # input
            imgs = sample['image']
            masks = sample['mask']
            imgs = imgs.cuda()
            masks_npy = np.array(masks.clone().detach())
            masks = masks.cuda()
            # train
            lr = scheduler(optimizer, i_batch, epoch)
            if cfg.model == 'unet':
                preds = model(imgs)
                preds = F.interpolate(preds, size=(masks.size(1), masks.size(2)), mode='bilinear')
                loss = criterion(preds, masks)
            else:
                preds, teacher_preds, pseudo = model(imgs, masks)
                preds = F.interpolate(preds, size=(masks.size(1), masks.size(2)), mode='bilinear')
                teacher_preds = F.interpolate(teacher_preds, size=(masks.size(1), masks.size(2)), mode='bilinear')
                loss = criterion(preds, masks, teacher_preds, pseudo, epoch)
            if distributed:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss = loss / acc_step
                loss.backward()
        
                if i_batch%acc_step == 0 or i_batch==len(dataset_train)-1:
                    optimizer.step()
                    optimizer.zero_grad()

            # output
            train_loss += loss.item()
            outputs = preds.cpu().detach().numpy()
            predictions = np.argmax(outputs, axis=1)
            metrics.update(masks_npy, predictions)
            scores_train = metrics.get_scores()

            batch_time.update(time.time()-start_time)
            start_time = time.time()
            if i_batch % 50 == 1 and local_rank == 0:
                tbar.set_description('Train loss: %.4f; mIoU: %.4f; batch time: %.2f' % 
                            (train_loss / (i_batch + 1), scores_train["mIoU"], batch_time.avg))
            # break
        metrics.reset()
        batch_time.reset() 

        ### Evaluation
        if evaluation and epoch % 5 == 4 and local_rank == 0:
            with torch.no_grad():
                model.eval()
                # Coarse
                print("evaluating coarse...")
                num_slides = len(dataset_coarse.slides)
                tbar2 = tqdm(range(num_slides))
                start_time = time.time()
                for i in tbar2:
                    pred = evaluator.val(dataset_coarse, model, i)
                    scores_coarse = evaluator.get_scores()
                    batch_time.update(time.time()-start_time)
                    tbar2.set_description('coarse mIoU: %.4f; slide time: %.2f' % 
                                            (scores_coarse["mIoU"], batch_time.avg))
                    # writer_info.update(mask=mask_rgb, prediction=predictions_rgb)
                    start_time = time.time()
                    # break
                batch_time.reset()
                scores_coarse = evaluator.get_scores()
                evaluator.reset_metrics()

                # Fine
                print("evaluating fine...")
                num_slides = len(dataset_fine.slides)
                tbar3 = tqdm(range(num_slides))
                start_time = time.time()
                for i in tbar3:
                    pred = evaluator.val(dataset_fine, model, i)
                    scores_fine = evaluator.get_scores()
                    batch_time.update(time.time()-start_time)
                    tbar3.set_description('fine mIoU: %.4f; slide time: %.2f' % 
                                            (scores_fine["mIoU"], batch_time.avg))
                    # writer_info.update(mask=mask_rgb, prediction=predictions_rgb)
                    start_time = time.time()
                    # break
                    
                batch_time.reset()
                scores_fine = evaluator.get_scores()
                evaluator.reset_metrics()

                # Save Model
                best_pred_fine, best_epoch = save_ckpt_model(model, cfg, scores_fine['mIoU'], best_pred_fine, best_epoch, epoch)
                # Log
                update_log(f_log, cfg, scores_train, scores_coarse, epoch, 
                            scores_fine=scores_fine)   
                log = '\n=>Epoches %i, best fine = %.4f, best epoch = %i \n\n' % (epoch, best_pred_fine, best_epoch)
                print(log)
                f_log.write(log)
                f_log.flush()
                # Writer  
                writer_info.update(
                        # lr=optimizer.param_groups[0]['lr'],
                        loss=train_loss/len(tbar),
                        train_mIoU=scores_train['mIoU'],
                        coarse_mIoU=scores_coarse['mIoU'],
                        fine_mIoU=scores_fine['mIoU'], 
                )
                update_writer(writer, writer_info, epoch)
        # break
    if local_rank == 0:     
        f_log.close() 

    # Generate best results based best epoch
    print("Output prediction results based on best model")
    ckpt_path = os.path.join(cfg.model_path, "%s-best-fine.pth"%(cfg.model+'-'+cfg.encoder))
    model = model_load_state_dict(model, ckpt_path=ckpt_path, distributed=distributed)
    if local_rank == 0:
        with torch.no_grad():
            num_slides = len(dataset_coarse.slides)
            for i in range(num_slides):
                pred_rgb = evaluator.inference(dataset_coarse, model, i)
                pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(cfg.coarse_output_path, dataset_coarse.slide+'.png'), pred_rgb)
                # break
            num_slides = len(dataset_fine.slides)
            for i in range(num_slides):
                pred_rgb = evaluator.inference(dataset_fine, model, i)
                pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(cfg.fine_output_path, dataset_fine.slide+'.png'), pred_rgb)
                # break


class SlideInference(object):
    def __init__(self, n_class, num_workers, batch_size, cfg):
        self.n_class = n_class
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.metrics = ConfusionMatrix(n_class)
        self.cfg = cfg
    
    def get_scores(self):
        scores = self.metrics.get_scores()
        return scores
    
    def reset_metrics(self):
        self.metrics.reset()
    
    def val(self, dataset, model, inds):
        dataset.get_patches_from_index(inds)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
        output = np.zeros((self.n_class, dataset.slide_size[0], dataset.slide_size[1])) # n_class x H x W
        template = np.zeros(dataset.slide_size, dtype='uint8') # H x W
        step = dataset.slide_step

        for sample in dataloader:
            imgs = sample['image']
            masks = sample['mask']
            coord = sample['coord']
            with torch.no_grad():
                imgs = imgs.cuda()
                masks = masks.cuda()
                if self.cfg.model == "unet":
                    preds = model.forward(imgs)
                else:
                    preds, teacher_preds, pseudo = model.forward(imgs, masks)
                preds = F.interpolate(preds, size=(imgs.size(2), imgs.size(3)), mode='bilinear')
                preds_np = preds.cpu().detach().numpy()
            
            _, _, h, w = preds_np.shape
            for i in range(imgs.shape[0]):
                x = math.floor(coord[0][i] * step[0])
                y = math.floor(coord[1][i] * step[1])
                output[:, x:x+h, y:y+w] += preds_np[i]
                template[x:x+h, y:y+w] += np.ones((h, w), dtype='uint8')
    
        template[template==0] = 1
        output = output / template
        prediction = np.argmax(output, axis=0)
        slide_mask = dataset.get_slide_mask_from_index(inds)
        self.metrics.update(slide_mask, prediction)

        return class_to_RGB(prediction)


    def inference(self, dataset, model, inds):
        dataset.get_patches_from_index(inds)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
        output = np.zeros((self.n_class, dataset.slide_size[0], dataset.slide_size[1])) # n_class x H x W
        template = np.zeros(dataset.slide_size, dtype='uint8') # H x W
        step = dataset.slide_step

        for sample in dataloader:
            imgs = sample['image']
            coord = sample['coord']
            with torch.no_grad():
                imgs = imgs.cuda()
                if self.cfg.model == "unet":
                    preds = model(imgs)
                else:
                    preds = model.inference(imgs)
                preds = F.interpolate(preds, size=(imgs.size(2), imgs.size(3)), mode='bilinear')
                preds_np = preds.cpu().detach().numpy()
            _, _, h, w = preds_np.shape
            for i in range(imgs.shape[0]):
                x = math.floor(coord[0][i] * step[0])
                y = math.floor(coord[1][i] * step[1])
                output[:, x:x+h, y:y+w] += preds_np[i]
                template[x:x+h, y:y+w] += np.ones((h, w), dtype='uint8')
    
        template[template==0] = 1
        output = output / template
        prediction = np.argmax(output, axis=0)

        return class_to_RGB(prediction)


def update_log(f_log, cfg, scores_train, scores_coarse, epoch, scores_fine=None):
    log = ""
    log = log + 'epoch [{}/{}] mIoU: train = {:.4f}, coarse = {:.4f}, fine = {:.4f}'.format(epoch, cfg.num_epochs, scores_train['mIoU'], scores_coarse['mIoU'], scores_fine['mIoU']) + "\n"
    
    log = log + "   Seg metric   \n"
    log = log + "    [train] IoU = " + str(scores_train['IoU']) + "\n"
    log = log + "    [train] Accuracy_mean = " + str(scores_train['mAcc'])  + "\n"
    log = log + "    [train] Precision = " + str(scores_train['Precision']) + "\n"
    log = log + "    [train] Recall = " + str(scores_train['Recall']) + "\n"
    log = log + "    ------------------------------------ \n"
    log = log + "    [coarse] IoU = " + str(scores_coarse['IoU']) + "\n"
    log = log + "    [coarse] Accuracy_mean = " + str(scores_coarse['mAcc'])  + "\n"
    log = log + "    [coarse] Precision = " + str(scores_coarse['Precision']) + "\n"
    log = log + "    [coarse] Recall = " + str(scores_coarse['Recall']) + "\n"
    if scores_fine:
        log = log + "    ------------------------------------ \n"
        log = log + "    [fine] IoU = " + str(scores_fine['IoU']) + "\n"
        log = log + "    [fine] Accuracy_mean = " + str(scores_fine['mAcc'])  + "\n"
        log = log + "    [fine] Precision = " + str(scores_fine['Precision']) + "\n"
        log = log + "    [fine] Recall = " + str(scores_fine['Recall']) + "\n"
    
    log += "================================\n"
    print(log)
    f_log.write(log)
    f_log.flush()


if __name__ == '__main__':
    from configs.remote_config_unet import Config

    args = argParser()
    cfg = Config(train=True)
    main(cfg, device, local_rank=local_rank)
    print(cfg.task_name)
    # w_list = [0, 0.5, 1]
    # for w in w_list:
    #     cfg = Config(train=True)
    #     cfg.loss_cfg[cfg.loss]['w'] = w
    #     main(cfg, device, local_rank=local_rank)
    

















