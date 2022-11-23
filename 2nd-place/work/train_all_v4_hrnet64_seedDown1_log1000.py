from os.path import expanduser

from common_utils_v4 import *
from config_main_v4 import cfg as main_cfg
import train_v4


@click.command()
@click.option("--gpu", type=str,
              # default="0",
              default="1",
              )
@click.option("--version", type=str, default="v4")
@click.option("--output_dir", type=str,
              # default="~/tmp/ddm_v4_outputs",
              # default="~/tmp/ddm_rerun221105b",
              default="~/tmp/ddm_rerun221106d_log1000",

              )
# @click.option("--debug", type=bool, default=True)  # todo <-- dbg ON/OFF
@click.option("--debug", type=bool, default=False)  # todo <-- dbg ON/OFF
def main(gpu, debug, version, output_dir):
    set_dbg(debug)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    main_cfg.MODELS = [
        ('stage3c/302val_cls3_normM', 'hrnet_w64', '3'),
        ('stage3c/313val_flatM_normT_mix01', 'hrnet_w64', '6'),
        ('stage3c/307val_cls3_normT_mix01', 'dpn107', '5'),
        ('stage3c/304val_cls3_normM_mix01', 'hrnet_w64', '8'),
        ('stage3c/303val_cls3_normT', 'hrnet_w32', '5'),
        ('stage3c/302val_cls3_normM', 'regnety_040', '4'),
        ('stage3c/307val_cls3_normT_mix01', 'dpn98', '9'),
        ('stage3c/304val_cls3_normM_mix01', 'hrnet_w32', '5'),
        ('stage3c/313val_flatM_normT_mix01', 'hrnet_w64', '7'),
        ('stage3c/302val_cls3_normM', 'hrnet_w64', '7'),
        ('stage3c/313val_flatM_normT_mix01', 'regnetx_320', '5'),
        ('stage3c/311val_flatM_mix01', 'hrnet_w64', '7'),
    ]

    for run_info in main_cfg.MODELS:
        cfg_name, timm_name, seed = run_info
        seed = int(seed)  # could be int('1')

        seed -= 1  # TODO <-------------------- note
        # timm_name = 'resnet34'
        timm_name = 'hrnet_w64'

        for fold in main_cfg.folds:
            print(f'fold={fold}, timm_name={timm_name}')
            cfg = load_config_data(
                cfg_name=cfg_name, main_cfg=main_cfg,
                timm_name=timm_name, fold=fold, seed=seed, version=version)
            cfg.OUTPUT_DIR = expanduser(output_dir)

            # making sure all set
            cfg.dataset_params.scale_before_log = 1000

            train_v4.train_and_predict(cfg=cfg)


if __name__ == "__main__":
    main()
