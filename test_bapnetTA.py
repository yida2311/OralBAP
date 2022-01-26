import os
import argparse
import time
import cv2
import math
import joblib
import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm
from torch.utils.data import DataLoader 

from dataset import OralDataset, OralSlide, Transformer, TransformerVal
from models import BAPnet, BAPnetTA
from utils.metric import ConfusionMatrix, AverageMeter
from utils.state_dict import model_Single2Parallel, save_ckpt_model, model_load_state_dict
from utils.vis_util import class_to_RGB, RGB_mapping_to_class
from utils.util import seed_everything, argParser, update_writer

# from train_bapnet import SlideInference


def main_slide(cfg, device):
    print(cfg.task_name)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    if not os.path.exists(cfg.test_output_path): 
        os.makedirs(cfg.test_output_path)

    ### MODEL INIT
    if cfg.model == 'bapnet':
        model = BAPnet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
    elif cfg.model == 'bapnetTA':
        model = BAPnetTA(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.modelTA_cfg)
    model = model_load_state_dict(model, ckpt_path=cfg.ckpt_path)
    model.to(device)
    ### DATASET
    dataset = OralSlide(
        cfg.testset_cfg["slide_list"],
        cfg.testset_cfg["img_dir"],
        cfg.testset_cfg["mask_dir"],
        cfg.testset_cfg["meta_file"],
        slide_mask_dir=cfg.testset_cfg["slide_mask_dir"],
        label=cfg.testset_cfg["label"], 
        transform=TransformerVal()
    )
    ### SOLVER
    batch_time = AverageMeter("BatchTime", ':3.3f')
    print("start testing......")
    ### LOG INIT
    f_log = open(cfg.log_path + cfg.task_name + ".log", 'w')
    log = cfg.task_name + '\n'
    for k, v in cfg.__dict__.items():
        log += str(k) + ' = ' + str(v) + '\n'
    print(log+'\n')
    f_log.write(log)
    f_log.flush()
    ### Running
    evaluator = SlideInference(cfg.n_class, cfg.num_workers, cfg.batch_size)
    #===========================================================
    with torch.no_grad():
        model.eval()
        num_slides = len(dataset.slides)
        tbar = tqdm(range(num_slides))
        start_time = time.time()
        for i in tbar:
            pred_rgb = evaluator.inference(dataset, model, i)
            pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(cfg.coarse_output_path, dataset.slide+'.png'), pred_rgb)
            seg_scores, cls_scores = evaluator.get_scores()
            batch_time.update(time.time()-start_time)
            log = 'slide: %s, mIoU: %.4f; mF1: %.4f; slide time: %.2f \n' % (dataset.slide, seg_scores["mIoU"], cls_scores['mF1'], batch_time.avg)
            tbar.set_description(log)
            f_log.write(log)
            f_log.flush()

            start_time = time.time() 
        
        seg_scores, cls_scores = evaluator.get_scores()
        log = "================== Overall Resutls ====================\n"
        log += "mIoU = {:.4f} \n".format(seg_scores['mIoU'])
        log = log + "IoU = " + str(seg_scores['IoU']) + "\n"
        log = log + "Accuracy_mean = " + str(seg_scores['mAcc'])  + "\n"
        log = log + "Precision = " + str(seg_scores['Precision']) + "\n"
        log = log + "Recall = " + str(seg_scores['Recall']) + "\n"
        log = log + "\n mF1 = {:.4f} \n".format(cls_scores['mF1'])
        log = log + "F1 = " + str(cls_scores['F1']) + "\n"
        print(log)
        f_log.write(log)
        f_log.flush()
        f_log.close() 


def main_patch(cfg, device):
    print(cfg.task_name)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    if not os.path.exists(cfg.test_output_path): 
        os.makedirs(cfg.test_output_path)
    if not os.path.exists(cfg.test_sim_path): 
        os.makedirs(cfg.test_sim_path)
    if not os.path.exists(cfg.test_pseudo_path): 
        os.makedirs(cfg.test_pseudo_path)

    ### MODEL INIT
    if cfg.model == 'bapnet':
        model = BAPnet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
    elif cfg.model == 'bapnetTA':
        model = BAPnetTA(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.modelTA_cfg)
    model = model_load_state_dict(model, ckpt_path=cfg.ckpt_path)
    model.to(device)
    ### DATASET
    dataset = OralSlide(
        cfg.testset_cfg["slide_list"],
        cfg.testset_cfg["img_dir"],
        cfg.testset_cfg["mask_dir"],
        cfg.testset_cfg["meta_file"],
        slide_mask_dir=cfg.testset_cfg["slide_mask_dir"],
        label=cfg.testset_cfg["label"], 
        transform=TransformerVal()
    )
    ### SOLVER
    batch_time = AverageMeter("BatchTime", ':3.3f')
    seg_metrics = ConfusionMatrix(cfg.n_class)
    print("start testing......")
    ### LOG INIT
    # f_log = open(cfg.log_path + cfg.task_name + ".log", 'w')
    # log = cfg.task_name + '\n'
    # for k, v in cfg.__dict__.items():
    #     log += str(k) + ' = ' + str(v) + '\n'
    # print(log+'\n')
    # f_log.write(log)
    # f_log.flush()
    ### Running
    #===========================================================
    with torch.no_grad():
        model.eval()
        num_slides = len(dataset.slides)
        tbar = tqdm(range(num_slides))
        start_time = time.time()
        for i in tbar:
            dataset.get_patches_from_index(i)
            slide = dataset.slide 
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
            print(slide)
            sim_save_path = os.path.join(cfg.test_sim_path, slide)
            pseudo_save_path = os.path.join(cfg.test_pseudo_path, slide)
            output_save_path = os.path.join(cfg.test_output_path, slide)
            if not os.path.exists(sim_save_path):
                os.makedirs(sim_save_path)
                os.makedirs(pseudo_save_path)
                os.makedirs(output_save_path)
            
            for sample in dataloader:
                imgs = sample['image']
                masks = sample['mask']
                masks_np = masks.numpy()
                name = sample['id']
                imgs = imgs.cuda()
                masks = masks.cuda()
                seg_feat, pseudo_label, _, _, sim, _ = model(imgs, masks)
                # for i in range(imgs.size(0)):
                #     sim[i] = sim[i] / sim[i].max()
                
                sim = np.array(255*sim.cpu().detach().numpy(), dtype='uint8')
                pseudo = pseudo_label.cpu().detach().numpy()
                output = np.argmax(seg_feat.cpu().detach().numpy(), axis=1)
                # save
                for i in range(imgs.size(0)):
                    sim_save_name = os.path.join(sim_save_path, name[i]+'.png')
                    cv2.imwrite(sim_save_name, sim[i])
                    pseudo_save_name = os.path.join(pseudo_save_path, name[i]+'.png')
                    pseudo_rgb = class_to_RGB(pseudo[i])
                    pseudo_rgb = cv2.cvtColor(pseudo_rgb, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(pseudo_save_name, pseudo_rgb)
                    output_save_name = os.path.join(output_save_path, name[i]+'.png')
                    output_rgb = class_to_RGB(output[i])
                    output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(output_save_name, output_rgb)
            
        #     pred_rgb = evaluator.inference(dataset, model, i)
        #     pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite(os.path.join(cfg.coarse_output_path, dataset.slide+'.png'), pred_rgb)
        #     seg_scores, cls_scores = evaluator.get_scores()
        #     batch_time.update(time.time()-start_time)
        #     log = 'slide: %s, mIoU: %.4f; mF1: %.4f; slide time: %.2f \n' % (dataset.slide, seg_scores["mIoU"], cls_scores['mF1'], batch_time.avg)
        #     tbar.set_description(log)
        #     f_log.write(log)
        #     f_log.flush()

        #     start_time = time.time() 
        
        # seg_scores, cls_scores = evaluator.get_scores()
        # log = "================== Overall Resutls ====================\n"
        # log += "mIoU = {:.4f} \n".format(seg_scores['mIoU'])
        # log = log + "IoU = " + str(seg_scores['IoU']) + "\n"
        # log = log + "Accuracy_mean = " + str(seg_scores['mAcc'])  + "\n"
        # log = log + "Precision = " + str(seg_scores['Precision']) + "\n"
        # log = log + "Recall = " + str(seg_scores['Recall']) + "\n"
        # log = log + "\n mF1 = {:.4f} \n".format(cls_scores['mF1'])
        # log = log + "F1 = " + str(cls_scores['F1']) + "\n"
        # print(log)
        # f_log.write(log)
        # f_log.flush()
        # f_log.close() 


def similarity_map_analyze(cfg, device):
    print(cfg.task_name)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    if not os.path.exists(cfg.test_sim_path): 
        os.makedirs(cfg.test_sim_path)
    if not os.path.exists(cfg.test_pseudo_path): 
        os.makedirs(cfg.test_pseudo_path)

    ### MODEL INIT
    model = BAPnet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
    model = model_load_state_dict(model, ckpt_path=cfg.ckpt_path)
    model.to(device)
    ### DATASET
    dataset = OralSlide(
        cfg.testset_cfg["slide_list"],
        cfg.testset_cfg["img_dir"],
        cfg.testset_cfg["mask_dir"],
        cfg.testset_cfg["meta_file"],
        slide_mask_dir=cfg.testset_cfg["slide_mask_dir"],
        label=cfg.testset_cfg["label"], 
        transform=TransformerVal()
    )

    batch_time = AverageMeter("BatchTime", ':3.3f')
    seg_metrics = ConfusionMatrix(cfg.n_class)
    print("start testing......")
    ### LOG INIT
    f_log = open(cfg.log_path + cfg.task_name + ".log", 'w')
    log = cfg.task_name + '\n'
    for k, v in cfg.__dict__.items():
        log += str(k) + ' = ' + str(v) + '\n'
    print(log+'\n')
    f_log.write(log)
    f_log.flush()

    with torch.no_grad():
        model.eval()
        num_slides = len(dataset.slides)
        tbar = tqdm(range(num_slides))
        start_time = time.time()
        for i in tbar:
            dataset.get_patches_from_index(i)
            slide = dataset.slide
            dataloader = DataLoader(dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
            # dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False, pin_memory=True)
            sim_save_path = os.path.join(cfg.test_sim_path, slide)
            pseudo_save_path = os.path.join(cfg.test_pseudo_path, slide)
            if not os.path.exists(sim_save_path):
                os.makedirs(sim_save_path)
                os.makedirs(pseudo_save_path)
            
            for sample in dataloader:
                imgs = sample['image']
                masks = sample['mask']
                masks_np = masks.numpy()
                name = sample['id']
                imgs = imgs.cuda()
                masks = masks.cuda()
                sim, pseudo = model.get_similarity_map(imgs, masks)
                sim = F.interpolate(sim.unsqueeze(1), size=(imgs.size(2), imgs.size(3)), mode='bilinear').squeeze(1)
                # pseudo = F.interpolate(pseudo, size=(imgs.size(2), imgs.size(3)), model='nearest')
                print(sim.max(), sim.min())
                sim = sim / sim.max()
                sim = np.array(255*sim.cpu().detach().numpy(), dtype='uint8')
                pseudo = pseudo.cpu().detach().numpy()
                seg_metrics.update(masks_np, pseudo)

                ## save
                for i in range(imgs.size(0)):
                    sim_save_name = os.path.join(sim_save_path, name[i]+'.png')
                    cv2.imwrite(sim_save_name, sim[i])
                    
                    pseudo_save_name = os.path.join(pseudo_save_path, name[i]+'.png')
                    pseudo_rgb = class_to_RGB(pseudo[i])
                    pseudo_rgb = cv2.cvtColor(pseudo_rgb, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(pseudo_save_name, pseudo_rgb)
    
    scores = seg_metrics.get_scores()
    log = "\n========= Overall Sim Results ===============\n"
    log += "mIoU = {:.4f} \n".format(scores['mIoU'])
    log = log + "IoU = " + str(scores['IoU']) + "\n"
    print(log)
    f_log.write(log)
    f_log.flush()
    f_log.close()


if __name__ == '__main__':
    # from configs_test.config_baseline_cs import Config
    from configs_test.config_baseline_mt_cs100 import Config

    cfg = Config(train=False)
    device = torch.device("cuda:0")
    # main(cfg, device)
    # similarity_map_analyze(cfg, device)
    main_patch(cfg, device)