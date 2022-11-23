import os
from default_config import basic_cfg


cfg = basic_cfg

cfg.name = os.path.basename(__file__).split(".")[0]
cfg.out_dir = "/data2/weights/seq_mars/seq/" + cfg.name #+'v1'

cfg.folds = [0,1,2,3,4]
cfg.train_bs = 8
cfg.seq_size = 700
cfg.max_mass = 350
cfg.bin_size = 2
cfg.seed = 359
cfg.epochs = 20
cfg.model = 'model13'
cfg.cache_path = 'cache/train_bin005.npy'

# cfg.normalize = 0
# cfg.use_meta = 1

cfg.feat_size = 350

# cfg.backbone = 'tf_efficientnet_b6_ns' #resnest50d_1s4x24d tf_efficientnet_b2_ns
cfg.drop_rate = 0.5
cfg.drop_path_rate = 0.1
# cfg.in_ch = 3
cfg.aug = 1
cfg.aug_prob = 0.6
cfg.mixup_prob = 0.6
cfg.is_sed = 1
cfg.loss_type = 'focal'
cfg.lr = 3e-6
cfg.use_val = 1
cfg.warmup = 400
# cfg.scale_aug = 1
# cfg.use_swa = 1
# cfg.round = 8
cfg.load_weight = '/data2/weights/seq_mars/seq/m13_bin005_r4/'