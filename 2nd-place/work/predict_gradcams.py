import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

from os.path import expanduser, dirname

import numpy as np
import cv2
from common_utils_v4 import *
from config_main_v4 import cfg as main_cfg
import train_v4
import common_utils_v4 as common_utils
from common_utils_v4 import *
import dataset_v4 as my_dataset
import models_v4 as models_cls
from config_main_v4 import cfg as main_cfg

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import *


@click.command()
@click.option("--gpu", type=str, default="0")
@click.option("--version", type=str, default="v4")
@click.option("--output_dir", type=str,

              # default='~/tmp/ddm_rerun221104',  # rerun
              # default="~/tmp/ddm_rerun221105", #/
              default='~/tmp/ddm_rerun221105b',  # rerun

              # default="~/tmp/ddm_v4_outputs",  #/tmp/ddm_v4_outputs_rerun221104
              # default="~/tmp/ddm_v4_outputs_rerun221104",  #/tmp/
              # default="~/tmp/ddm_v4b",  #/tmp/
              )
@click.option("--dataset_type", type=str,
              # default="test_val",   #  ["train", "val", "orig_val", "test_val"]
              # default="orig_val",   #  ["train", "val", "orig_val", "test_val"]
              # default="all",   #  ["train", "val", "orig_val", "test_val"]
              default="val",  # ["train", "val", "orig_val", "test_val"]
              )
@click.option("--debug", type=bool,
              default=True,
              # default=False,
              )
def main(gpu, debug, version, output_dir, dataset_type):
    set_dbg(debug)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    assert dataset_type in ["all", "val", "orig_val", "test_val"]
    run_datasets = [dataset_type]
    if dataset_type == 'all':
        run_datasets = ["val", "test_val"]

    # Making sure all report/write-up/debug options are off
    main_cfg.SORT_SAVE_SAMPLES = False
    main_cfg.SHOW_SAMPLES = False
    main_cfg.EXPORT_MAX_MASS_1D = False

    # DBG:
    main_cfg.MODELS = [('stage3a/313val_flatM_normT_mix01', 'hrnet_w64', '4', 'loss_val')]

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
                    predict_gradcams(cfg=cfg, dataset_type=dataset_type, best_tag=best_tag)
            # except Exception as e:
            #     print(e)
            #     continue


def predict_gradcams(*,
                     cfg,
                     dataset_type,  # "val", "orig_val", "test_val"
                     best_tag,  # 'val_loss', 'val_neg_acc'
                     ):
    seed_everything(cfg.seed)

    load_dir_names(cfg)

    # Set to ONE image
    cfg.train_data_loader.batch_size = 1
    cfg.train_data_loader.num_workers = 0
    cfg.batch_size = 1

    data_loaders = my_dataset.make_loaders(cfg)
    model = train_v4.build_model(cfg, pretrained=False)
    print(model.__class__.__name__)

    train_v4.load_checkpoint(cfg=cfg, model=model, best_tag=best_tag)
    model = model.cuda()
    model.eval()

    # incre_modules: for hrnet_w64
    generate_cam = GradCAMPlusPlus(
        model=model, target_layers=[model.base_model.incre_modules[-1]], use_cuda=True)

    # for phase in ["train", "val", "orig_val", "test_val"]:
    phase = dataset_type
    data_loader = data_loaders[phase]
    y_true = []
    y_pred = []
    data_iter = tqdm(data_loader, disable=False)
    # saved = set_dbg(False)
    for data in data_iter:
        img = data['image'].float().cuda()
        targets = data['label'][0]
        if targets.sum() != 1:  # only plot one target
            continue

        with torch.set_grad_enabled(False):
            image = data['image'].float().cuda()
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                pred = model(image)

        y_pred = torch.sigmoid(pred).detach().cpu().numpy()[0]
        y_pred = (y_pred > 0.9).astype(int)  # GOOD prediction
        y_true = targets.detach().cpu().numpy().astype(int)
        if np.sum(y_pred != y_true) > 0:
            continue  # ignore if wrong

        cam = generate_cam(
            input_tensor=img,
            # targets=[targets],
            aug_smooth=True,
            eigen_smooth=True)
        cam = cam[0, :]

        x = img.detach().cpu().numpy()
        dbg(x)
        x = np.moveaxis(x[0], source=0, destination=-1)
        dbg(x)
        if len(x.shape) == 3:
            x = np.sum(x, axis=-1, keepdims=True)
        else:
            x = x[:, :, None]
        x = np.repeat(x, repeats=3, axis=-1)
        x = x / x.max()
        x = x.astype(np.float32)
        cam_img = show_cam_on_image(img=x, mask=cam, use_rgb=True)

        img_id = data['sample_id'][0]
        img_label = [str(int(i)) for i in targets]
        img_label = "".join(img_label)

        cam_img = np.flipud(cam_img)
        # cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
        tag = f'{img_id}_{img_label}'
        fpath = expanduser(f'~/tmp/ddm/dbg_grad_cams/{tag}.png')
        os.makedirs(dirname(fpath), exist_ok=True)
        cv2.imwrite(fpath, cam_img)

        plt.imshow(cam_img)
        # plt.savefig(fpath)
        plt.show()

        # y_true.append(data['label'].float().numpy())
        # y_pred.append(torch.sigmoid(pred).detach().cpu().numpy())

    # set_dbg(saved)


# def get_grad_cam(model):
#     # FROM:
#     # https://github.com/drivendataorg/wheres-whale-do/tree/main/Explainability%20Bonus/4th%20place


if __name__ == "__main__":
    main()
