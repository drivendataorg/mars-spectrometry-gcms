import os
from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.seed = 42
cfg.epochs = 13
cfg.folds = [0, 1, 2, 3, 4]  
cfg.seq_size = 2000
cfg.max_mass = 500
cfg.bin_size = 2
cfg.cache_path = 'cache/cache.npy'
cfg.use_swa = False 
cfg.loss_type = 'bce'
cfg.model = 'model3'
cfg.normalize = 1
cfg.in_ch = 1
cfg.aug = 0
cfg.aug_prob = 1
cfg.mixup_prob = 0.
cfg.use_meta = 0
cfg.power_r = 2.5
cfg.power_prob = 0
cfg.is_pretrain = 0
cfg.load_weight = ''
cfg.lr = 1e-4
cfg.feat_size = 252
cfg.use_val = 0
cfg.all_data = 0
cfg.is_sed = 0
cfg.stride = 1
cfg.num_folds = 5
cfg.round = 1
cfg.split = 1
cfg.rot = 0
cfg.scale_aug = 0
cfg.warmup = 200
cfg.target_cols = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']

basic_cfg = cfg