import os
import argparse
import time
import cv2
import math
import joblib
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm
from torch.utils.data import DataLoader 

from dataset import OralDataset, OralSlide, Transformer, TransformerVal
from models import Unet
from utils.metric import ConfusionMatrix, AverageMeter
from utils.state_dict import model_Single2Parallel, model_load_state_dict
from utils.vis_util import class_to_RGB, RGB_mapping_to_class

from train_unet import SlideInference


def main(cfg, device):
    print(cfg.task_name)
    if not os.path.exists(cfg.log_path):
        os.makedirs(cfg.log_path)
    if not os.path.exists(cfg.test_output_path): 
        os.makedirs(cfg.test_output_path)

    ### MODEL INIT
    model = Unet(classes=cfg.n_class, encoder_name=cfg.encoder, **cfg.model_cfg)
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
    single_metrics = ConfusionMatrix(cfg.n_class)
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
            dataset.get_patches_from_index(i)
            pred, pred_rgb, _ = evaluator.inference(dataset, model)
            
            label = dataset.get_slide_mask_from_index(i)
            evaluator.update_scores(label, pred)
            single_metrics.update(label, pred)
            scores = single_metrics.get_scores()
            single_metrics.reset()
            batch_time.update(time.time()-start_time)
            log = 'slide: %s, mIoU: %.4f; slide time: %.2f \n' % (dataset.slide, scores["mIoU"], batch_time.avg)
            tbar.set_description(log)
            f_log.write(log)
            f_log.flush()

            pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(cfg.test_output_path, dataset.slide+'.png'), pred_rgb)

            start_time = time.time() 
        
        scores = evaluator.get_scores()
        log = "================== Overall Resutls ====================\n"
        log += "mIoU = {:.4f} \n".format(scores['mIoU'])
        log = log + "IoU = " + str(scores['IoU']) + "\n"
        log = log + "Accuracy_mean = " + str(scores['mAcc'])  + "\n"
        log = log + "Precision = " + str(scores['Precision']) + "\n"
        log = log + "Recall = " + str(scores['Recall']) + "\n"
        print(log)
        f_log.write(log)
        f_log.flush()
        f_log.close() 


if __name__ == '__main__':
    from configs.config_unet import Config

    cfg = Config(train=False)
    device = torch.device("cuda:0")
    main(cfg, device)
