from utils.util import simple_time
import json

class Config:
    def __init__(self, train=True):
        # model config
        self.model = "unet"
        self.encoder = "resnet34"  
        self.n_class = 4
        self.model_cfg = {
            'encoder_depth': 5,
            'encoder_weights': 'imagenet',
            'decoder_use_batchnorm': True,
            'decoder_channels': (256, 128, 64, 64),
            'decoder_attention_type': 'scse',
            'in_channels': 3,
        }
        self.train = train

        # data config
        root = '/media/ldy/7E1CA94545711AE6/OSCC/5x_tile/5x_1000/'
        self.trainset_cfg = {
            "img_dir": root + "patch/",
            "mask_dir": root + "std_mask/",
            "meta_file": root + "train.csv",
            "label": True,
        }
        with open(root+'train_val_part.json', 'r') as f:
            coarse_slide_list = json.load(f)['val']
        self.coarseset_cfg = {
            "slide_list": coarse_slide_list,
            "img_dir": root +  "patch/",
            "mask_dir": root +  "std_mask/",
            "slide_mask_dir": "/media/ldy/7E1CA94545711AE6/OSCC/" + "5x_mask/std_mask/",
            "meta_file": root + "tile_info.json",
            "label": True,
        }

        fine_root = '/media/ldy/7E1CA94545711AE6/OSCC_FINE/full_anno/'
        with open(fine_root+'5x_tile/slide.json', 'r') as f:
            fine_slide_list = json.load(f)
        self.fineset_cfg = {
            "slide_list": fine_slide_list,
            "img_dir": fine_root +  "5x_tile/5x_800/patch/",
            "mask_dir": fine_root + "5x_tile/5x_800/std_mask/",
            "slide_mask_dir": fine_root + "std_mask_slide/",
            "meta_file": fine_root + "5x_tile/5x_800/tile_info.json",
            "label": True,
        }
        self.crop_size = 800

        # train config
        self.scheduler = 'poly' # ['cos', 'poly', 'step', 'ym']
        self.lr = 1e-4
        self.num_epochs = 120
        self.warmup_epochs = 2
        self.batch_size = 1
        self.acc_step = 4
        self.ckpt_path = None # pretrained model
        self.num_workers = 4
        self.evaluation = True  # evaluatie val set
        self.val_vis = True # val result visualization

        # loss config
        self.loss = "ce" # ["ce", "sce", 'ce-dice]
        self.loss_cfg = {
            "sce": {
                "alpha": 1.0,
                "beta": 1.0,
            },
            "ce-dice": {
                "alpha": 0.1,
            },
            "focal": {
                "gamma": 2,
            }

        }

        # task name
        self.task_name = "-".join([self.model, self.loss, simple_time()])
        if train:
            self.task_name += "-" + "train"
        else:
            self.task_name += "-" + "test"
        # output config
        out_root = "results/"
        self.model_path = out_root + "saved_models/" + self.task_name
        self.log_path = out_root + "logs/" 
        self.writer_path = out_root + 'writers/' + self.task_name
        self.coarse_output_path = out_root + "corse predictions/" + self.task_name 
        self.fine_output_path = out_root + "fine predictions/" + self.task_name 
        
        # test cfg
        self.testset_cfg = self.fineset_cfg
        self.test_output_path = out_root + "test predictions/" + self.task_name 

