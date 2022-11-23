import os
from default_config import basic_cfg


cfg = basic_cfg

cfg.name = os.path.basename(__file__).split(".")[0]
cfg.out_dir = "/data2/weights/seq_mars/seq/" + cfg.name #+ 'r2'

cfg.folds = [0,1,2,3,4]
cfg.train_bs = 6
cfg.seq_size = 700
cfg.max_mass = 400
cfg.bin_size = 2
cfg.seed = 49
cfg.epochs = 20
cfg.model = 'model_sed'
cfg.backbone = 'tf_efficientnet_b7_ns' #resnest50d_1s4x24d tf_efficientnet_b2_ns
cfg.drop_rate = 0.5
cfg.drop_path_rate = 0.1
cfg.in_ch = 3
cfg.aug = 1
cfg.mixup_prob = 0.5
cfg.is_sed = 1
cfg.loss_type = 'focal'
cfg.lr = 1e-4
cfg.use_val = 1
# cfg.use_swa = 1
cfg.load_weight = '/data2/weights/seq_mars/seq/b7_bin005_sed_3ch/'
cfg.cache_path = 'cache/train_bin005.npy'
