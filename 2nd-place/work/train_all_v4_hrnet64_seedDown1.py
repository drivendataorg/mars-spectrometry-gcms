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
              default="~/tmp/ddm_rerun221105b",
              )
# @click.option("--debug", type=bool, default=True)  # todo <-- dbg ON/OFF
@click.option("--debug", type=bool, default=False)  # todo <-- dbg ON/OFF
def main(gpu, debug, version, output_dir):
    set_dbg(debug)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

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

            train_v4.train_and_predict(cfg=cfg)


if __name__ == "__main__":
    main()
