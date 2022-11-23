from os.path import expanduser

from common_utils_v4 import *
from config_main_v4 import cfg as main_cfg
import train_v4

@click.command()
@click.option("--gpu", type=str, default="0")
@click.option("--version", type=str, default="v4")
@click.option("--output_dir", type=str,
              default='~/tmp/ddm_2nd_place_weights_dbg',  # rerun
              # default='~/tmp/ddm_2nd_place_weights',  # rerun
              )
@click.option("--dataset_type", type=str,
              # default="test_val",   #  ["train", "val", "orig_val", "test_val"]
              # default="orig_val",   #  ["train", "val", "orig_val", "test_val"]
              default="all",   #  ["train", "val", "orig_val", "test_val"]
              # default="val",   #  ["train", "val", "orig_val", "test_val"]
              )
@click.option("--debug", type=bool,
              # default=True,
              default=False,
              )
def main(gpu, debug, version, output_dir, dataset_type):
    set_dbg(debug)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    assert dataset_type in ["all", "val", "orig_val", "test_val"]
    run_datasets = [dataset_type]
    if dataset_type == 'all':
        # NOTE: val == oof
        run_datasets = ["val", "test_val"]

    # Making sure all report/write-up/debug options are off
    main_cfg.SORT_SAVE_SAMPLES = False
    main_cfg.SHOW_SAMPLES = False
    main_cfg.EXPORT_MAX_MASS_1D = False

    # DBG:
    # main_cfg.MODELS = [('stage3a/313val_flatM_normT_mix01', 'hrnet_w64', '4', 'loss_val')]

    for run_info in main_cfg.MODELS:
        if len(list(run_info)) == 3:
            cfg_name, timm_name, seed = run_info  # to run original submission
        else:
            cfg_name, timm_name, seed, best_tag = run_info
            main_cfg.save_oof_tags = [best_tag]  # it was set to run many but here only one is used

        seed = int(seed)  # could be int('1')
        for fold in main_cfg.folds:
            print(f'fold={fold}, timm_name={timm_name}')
            cfg = load_config_data(
                cfg_name=cfg_name, main_cfg=main_cfg,
                timm_name=timm_name, fold=fold, seed=seed, version=version)
            cfg.OUTPUT_DIR = expanduser(output_dir)
            # try:
            for dataset_type in run_datasets:
                for best_tag in main_cfg.save_oof_tags:
                    train_v4.predict(cfg=cfg, dataset_type=dataset_type, best_tag=best_tag)
            # except Exception as e:
            #     print(e)
            #     continue


if __name__ == "__main__":
    main()

