from utils.util import simple_time
import json

class Config:
    def __init__(self, train=True):
        # model config
        self.model = "bapnetMT"
        self.encoder = "resnet34"  
        self.n_class = 4
        self.model_cfg = {
            'encoder_depth': 5,
            'encoder_weights': 'imagenet',
            'decoder_use_batchnorm': True,
            'decoder_channels': (256, 128, 64, 64),
            'decoder_attention_type': 'scse',
            'in_channels': 3,
            "proj_channels": 128,
            'aux_params': {
                'memory_bank': {
                    'K': 1000,
                    'T': 100,
                    'm': 0.9,
                },
                'min_ratio': 0.1,
                'momentum': 0.9,
                'temperature': 1.0,
                'weight_type': 'weighted',  # 'softmax', 'weighted', 'mean'
            }
        }
        self.train = train

        # loss config
        self.loss = "bapMT" # ["ce", "sce", 'ce-dice]
        self.loss_cfg = {
            "bapMT": {
                "alpha": 1.0,
                "beta": 1.0,
                "w": 0.2,
                "use_curriculum": True,
                "use_seg_sim_cons": False,
            },
        }

        # task name
        self.task_name = "-".join([self.model, self.loss, simple_time()])
        # self.task_name = "-".join([self.model, self.loss, '[11-05]'])
        if train:
            self.task_name += "-" + "train-2"
        else:
            self.task_name += "-" + "test"

        # data config
        root = '/remote-home/share/ldy/OSCC/'
        train_root = root + '2.5x_tile/2.5x_640/'
        self.trainset_cfg = {
            "img_dir": train_root + "patch/",
            "mask_dir": train_root + "std_mask/",
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
        # test set
        with open(train_root+'train_coarse_fine.json', 'r') as f:
            train_slide_list = json.load(f)['train']
        self.testset_cfg = {
            "slide_list": train_slide_list,
            "img_dir": train_root +  "patch/",
            "mask_dir": train_root +  "std_mask/",
            "slide_mask_dir": root + "2.5x_mask/std_mask/",
            "meta_file":train_root + "tile_info.json",
            "label": True,
        }

        # train config
        self.scheduler = 'poly' # ['cos', 'poly', 'step', 'ym']
        self.lr = 1e-4
        self.num_epochs = 120
        self.warmup_epochs = 2
        self.batch_size = 12
        self.acc_step = 1
        self.ckpt_path = None
        if not train:
            self.ckpt_path = 'results/saved_models/bapnet-bap1-[11-13-15]-train/bapnet-resnet34-best-fine.pth' # pretrained model
        self.num_workers = 4
        self.evaluation = True  # evaluatie val set
        self.val_vis = True # val result visualization

        # output config
        out_root = "results-v2/"
        self.model_path = out_root + "saved_models/" + self.task_name
        self.log_path = out_root + "logs/" 
        self.writer_path = out_root + 'writers/' + self.task_name
        self.sim_output_path = out_root + 'sim/' + self.task_name
        self.sim_pseudo_output_path = out_root + 'sim pseudo/' + self.task_name
        self.seg_pseudo_output_path = out_root + 'seg pseudo/' + self.task_name
        self.coarse_output_path = out_root + "coarse predictions/" + self.task_name 
        self.fine_output_path = out_root + "fine predictions/" + self.task_name 
        
        # test cfg
        # self.testset_cfg = self.fineset_cfg
        self.test_output_path = out_root + "test predictions/" + self.task_name + '/output'
        self.test_sim_path = out_root + "test predictions/" + self.task_name +'/sim/'
        self.test_sim_pseudo_path = out_root + "test predictions/" + self.task_name +'/sim pseudo/'
        self.test_seg_pseudo_path = out_root + "test predictions/" + self.task_name +'/seg pseudo/'



