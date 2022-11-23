from os.path import expanduser, join
from types import SimpleNamespace

from common_utils_v4 import toDotDict

cfg = SimpleNamespace(**{})

# cfg.train_seed = None  # moved to args

# DATA_DIR = 'data'
cfg.DATA_DIR = expanduser('./data/raw')
# TRAIN_DIR = f'{DATA_DIR}/train_features'
cfg.TRAIN_DIR = join(cfg.DATA_DIR, 'train_features')
# OUTPUT_DIR = '../../mars_spectrometry_submission_output'
cfg.OUTPUT_DIR = expanduser('~/datavol/data/raw/winner-subs/2nd-place/tmp/ddm')
cfg.TMP_DIR = expanduser('~/datavol/data/tmp/ddm')

# cfg.features_pp_dir = join(cfg.DATA_DIR, 'preprocess_data_v1d_normT')
# cfg.features_pp_dir = join(cfg.DATA_DIR, 'preprocess_data_v3b_log1000')

# n_folds = 4
cfg.n_folds = 5
cfg.folds = [0, 1, 2, 3, 4]
# cfg.folds_csv_path = join(cfg.DATA_DIR, 'folds_n5_seed42_v1a.csv')
cfg.folds_csv_path = 'folds_n5_seed42_v1a.csv'
cfg.train_labels_path = join(cfg.DATA_DIR, 'train_labels.csv')
cfg.val_labels_path = join(cfg.DATA_DIR, 'val_labels.csv')

cfg.cls_labels = [
    # 'basalt', 'carbonate', 'chloride', 'iron_oxide', 'oxalate', 'oxychlorine', 'phyllosilicate', 'silicate', 'sulfate', 'sulfide'
    # sample_id,aromatic,hydrocarbon,carboxylic_acid,nitrogen_bearing_compound,chlorine_bearing_compound,sulfur_bearing_compound,alcohol,other_oxygen_bearing_compound,mineral
    # S0000,0,0,0,0,0,0,0,0,1
    'aromatic', 'hydrocarbon', 'carboxylic_acid', 'nitrogen_bearing_compound', 'chlorine_bearing_compound',
    'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound', 'mineral'
]

# axes
cfg.col_m = 'mass'  # was 'm/z'
cfg.col_t = 'time'  # was 'temp'
cfg.col_a = 'intensity'  # i.e. amplitude (signal strength); was 'abundance'

cfg.norm_max_a = 1
cfg.norm_max_t = 256 * 4
cfg.max_t = cfg.norm_max_t

# Helium ions will typically show up in the data as ions detected with an m/z value of 4.0 and are usually disregarded,
# along with most ions with mass values of less than 12.
# Most ions relevant to the detection of the label classes for this competition will be below 250,
# but there are some compounds in these experiments that will generate ions between 250 and 500.
cfg.max_m = 256  # todo maybe 250, 500
cfg.min_m = 12  # was 28
cfg.m_scale_for_image = 1

cfg.n_classes = len(cfg.cls_labels)
cfg.use_amp = False
# cfg.save_oof_tags = ['loss_val', 'oof_neg_acc']
cfg.save_oof_tags = ['loss_val']  # i.e. oof_loss

# # list of models used for submission, see oof0.08631_ddm_v4_outputs_test_mean_logits_clip1e4.py
# cfg.MODELS = [('stage3a/302val_cls3_normM', 'hrnet_w64', '3', 'loss_val'),
#  ('stage3a/302val_cls3_normM', 'hrnet_w64', '7', 'loss_val'),
#  ('stage3a/302val_cls3_normM', 'regnety_040', '4', 'loss_val'),
#  ('stage3a/303val_cls3_normT', 'hrnet_w32', '5', 'loss_val'),
#  ('stage3a/304val_cls3_normM_mix01', 'hrnet_w32', '5', 'loss_val'),
#  ('stage3a/304val_cls3_normM_mix01', 'hrnet_w64', '8', 'loss_val'),
#  ('stage3a/307val_cls3_normT_mix01', 'dpn107', '5', 'loss_val'),
#  ('stage3a/307val_cls3_normT_mix01', 'dpn98', '9', 'loss_val'),
#  ('stage3a/313val_flatM_normT_mix01', 'hrnet_w64', '6', 'loss_val'),
#  ('stage3a/313val_flatM_normT_mix01', 'hrnet_w64', '7', 'loss_val'),
#  ('stage3a/313val_flatM_normT_mix01', 'regnetx_320', '5', 'loss_val')]

# re-run: oof0.085937_ddm_rerun04_05b_06c_06d_06e_test_mean_logits_clip1e4.py
cfg.MODELS = [('stage3a/302val_cls3_normM', 'hrnet_w64', '3', 'loss_val'),
 ('stage3a/302val_cls3_normM', 'hrnet_w64', '6', 'loss_val'),
 ('stage3a/302val_cls3_normM', 'hrnet_w64', '7', 'loss_val'),
 ('stage3a/302val_cls3_normM', 'resnet34', '3', 'loss_val'),
 ('stage3a/304val_cls3_normM_mix01', 'hrnet_w64', '4', 'loss_val'),
 ('stage3a/304val_cls3_normM_mix01', 'resnet34', '5', 'loss_val'),
 ('stage3a/307val_cls3_normT_mix01', 'dpn107', '5', 'loss_val'),
 ('stage3a/307val_cls3_normT_mix01', 'hrnet_w64', '8', 'loss_val'),
 ('stage3a/313val_flatM_normT_mix01', 'hrnet_w64', '4', 'loss_val'),
 ('stage3a/313val_flatM_normT_mix01', 'regnetx_320', '5', 'loss_val'),
 ('stage3c/302val_cls3_normM', 'hrnet_w64', '7', 'loss_val'),
 ('stage3c/302val_cls3_normM', 'resnet34', '3', 'loss_val'),
 ('stage3c/303val_cls3_normT', 'hrnet_w32', '5', 'loss_val'),
 ('stage3c/307val_cls3_normT_mix01', 'dpn98', '9', 'loss_val'),
 ('stage3c/313val_flatM_normT_mix01', 'hrnet_w64', '5', 'loss_val'),
 ('stage3c/313val_flatM_normT_mix01', 'hrnet_w64', '6', 'loss_val')]

# generate plots for write-up/report
# cfg.SORT_SAVE_SAMPLES = True
# cfg.SHOW_SAMPLES = True
# cfg.EXPORT_MAX_MASS_1D = True
cfg.SORT_SAVE_SAMPLES = False
cfg.SHOW_SAMPLES = False
cfg.EXPORT_MAX_MASS_1D = False


cfg = vars(cfg)  # to dict: https://stackoverflow.com/questions/52783883/how-to-initialize-a-dict-from-a-simplenamespace
cfg = toDotDict(cfg)

# ORIG_VAL---------------------------------
