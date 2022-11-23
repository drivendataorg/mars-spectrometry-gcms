import os
from default_config import basic_cfg


cfg = basic_cfg

cfg.name = os.path.basename(__file__).split(".")[0]
cfg.out_dir = "/data2/weights/seq_mars/seq/" + cfg.name #+ 'r2'

cfg.folds = [0,1,2,3,4]
cfg.train_bs = 8
cfg.seq_size = 750
cfg.max_mass = 430
cfg.bin_size = 2
cfg.seed = 15
cfg.epochs = 20
cfg.model = 'model_sed'
cfg.backbone = 'tf_efficientnet_b5_ns' #resnest50d_1s4x24d tf_efficientnet_b2_ns
cfg.drop_rate = 0.5
cfg.drop_path_rate = 0.1
cfg.in_ch = 3
cfg.aug = 1
cfg.aug_prob = 0.6
cfg.mixup_prob = 0.6
cfg.is_sed = 1
cfg.loss_type = 'focal'
cfg.lr = 2e-5
cfg.use_val = 1
# cfg.use_swa = 1
cfg.load_weight = '/data2/weights/seq_mars/seq/b5_bin005_sed_3ch_v1_r5/'
cfg.cache_path = 'cache/train_bin005.npy'
