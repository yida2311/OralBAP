from utils.util import simple_time
import json

class Config:
    def __init__(self, train=True):
        # model config
        self.model = "unetWithCLS"
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
        root = '/media/ldy/7E1CA94545711AE6/OSCC/'
        train_root = root + '2.5x_tile/2.5x_640/'
        self.trainset_cfg = {
            "img_dir": train_root + "patch/",
            "mask_dir": train_root + "std_mask/",
            # "mask_dir": train_root + "std_mask_dilate50/",
            "meta_file": train_root + "train.csv",
            "label": True,
        }
        with open(train_root+'train_coarse_fine.json', 'r') as f:
            coarse_slide_list = json.load(f)['coarse']
        self.coarseset_cfg = {
            "slide_list": coarse_slide_list,
            "img_dir": train_root +  "patch/",
            "mask_dir": train_root +  "std_mask/",
            "slide_mask_dir": root + "2.5x_mask/std_mask/",
            "meta_file": train_root + "tile_info.json",
            "label": True,
        }

        with open(train_root+'train_coarse_fine.json', 'r') as f:
            fine_slide_list = json.load(f)['fine']
        self.fineset_cfg = {
            "slide_list": fine_slide_list,
            "img_dir": train_root +  "patch/",
            "mask_dir": train_root +  "std_mask/",
            "slide_mask_dir": root + "2.5x_mask_fine/std_mask/",
            "meta_file": train_root + "tile_info.json",
            "label": True,
        }
        self.crop_size = 513

        # train config
        self.scheduler = 'poly' # ['cos', 'poly', 'step', 'ym']
        self.lr = 1e-4
        self.num_epochs = 120
        self.warmup_epochs = 2
        self.batch_size = 12
        self.acc_step = 1
        self.ckpt_path = None # pretrained model
        if not train:
            self.ckpt_path = 'results/saved_models/bapnet-bap-[11-09-15]-train/bapnet-resnet34-best-fine.pth' # pretrained model
        self.num_workers = 4
        self.evaluation = True  # evaluatie val set
        self.val_vis = True # val result visualization

        # loss config
        self.loss = "unetWithCLS" # ["ce", "sce", 'ce-dice]
        self.loss_cfg = {
            "unetWithCLS": {
                "alpha": 1.0,
            },
        }

        # task name
        self.task_name = "-".join([self.model, self.loss, simple_time()])
        if train:
            self.task_name += "-" + "train"
            # self.task_name += "-dilate50-" + "train"
        else:
            self.task_name += "-" + "test"
        # output config
        out_root = "results/"
        self.model_path = out_root + "saved_models/" + self.task_name
        self.log_path = out_root + "logs/" 
        self.writer_path = out_root + 'writers/' + self.task_name
        self.coarse_output_path = out_root + "coarse predictions/" + self.task_name 
        self.fine_output_path = out_root + "fine predictions/" + self.task_name 
        
        # test cfg
        self.testset_cfg = self.fineset_cfg
        self.test_output_path = out_root + "test predictions/" + self.task_name 

