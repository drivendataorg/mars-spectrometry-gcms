# to run:
#>   source bash_scripts/train_all.sh


#cfg.MODELS = [('stage3a/302val_cls3_normM', 'hrnet_w64', '3', 'loss_val'),
 # ('stage3a/302val_cls3_normM', 'hrnet_w64', '6', 'loss_val'),
 # ('stage3a/302val_cls3_normM', 'hrnet_w64', '7', 'loss_val'),
 # ('stage3a/302val_cls3_normM', 'resnet34', '3', 'loss_val'),
 # ('stage3a/304val_cls3_normM_mix01', 'hrnet_w64', '4', 'loss_val'),
 # ('stage3a/304val_cls3_normM_mix01', 'hrnet_w64', '7', 'loss_val'),
 # ('stage3a/307val_cls3_normT_mix01', 'dpn107', '5', 'loss_val'),
 # ('stage3a/307val_cls3_normT_mix01', 'hrnet_w64', '8', 'loss_val'),
 # ('stage3a/313val_flatM_normT_mix01', 'hrnet_w64', '4', 'loss_val'),
 # ('stage3a/313val_flatM_normT_mix01', 'hrnet_w64', '6', 'loss_val'),
 # ('stage3a/313val_flatM_normT_mix01', 'regnetx_320', '5', 'loss_val'),
 # ('stage3c/302val_cls3_normM', 'resnet34', '3', 'loss_val'),
 # ('stage3c/307val_cls3_normT_mix01', 'resnet34', '5', 'loss_val')]

GPU="0"
OUTPUT_DIR="~/tmp/ddm_rerunYYMMDD"

#SEED=1
#main(action, exp, fold, seed, gpu, debug, version, timm_name, output_dir):

NAME="stage3a/302val_cls3_normM"
TIMM_NAME="hrnet_w64"
SEED=3
python train_v4.py --action train_all_folds\
 --exp $NAME\
 --timm_name $TIMM_NAME\
 --seed $SEED\
 --debug False --gpu $GPU\
 --output_dir $OUTPUT_DIR

SEED=6
python train_v4.py --action train_all_folds\
 --exp $NAME\
 --timm_name $TIMM_NAME\
 --seed $SEED\
 --debug False --gpu $GPU\
 --output_dir $OUTPUT_DIR

SEED=7
python train_v4.py --action train_all_folds\
 --exp $NAME\
 --timm_name $TIMM_NAME\
 --seed $SEED\
 --debug False --gpu $GPU\
 --output_dir $OUTPUT_DIR

TIMM_NAME="resnet34"
SEED=3
python train_v4.py --action train_all_folds\
 --exp $NAME\
 --timm_name $TIMM_NAME\
 --seed $SEED\
 --debug False --gpu $GPU\
 --output_dir $OUTPUT_DIR

