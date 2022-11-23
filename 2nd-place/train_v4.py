from os.path import expanduser

import common_utils_v4 as common_utils
from common_utils_v4 import *
import dataset_v4 as my_dataset
import models_v4 as models_cls
from config_main_v4 import cfg as main_cfg


@click.command()
@click.option("--action", type=str,
              default='train',
              # default='train_all_folds',
              )
@click.option("--exp", type=str,
              default='stage3b/311val_dbg',
              # default='stage3a/302val_cls3_normM',
              # default='stage3a/301val_cls3_m256t192_ep20tta32x5',
              # default='stage3a/303val_cls3_normT',
              # default='stage3a/304val_cls3_normM_mix01',
              # default='stage3a/305val_cls3_mix01',
              # default='stage3a/311val_flatM_mix01',
              # default='stage3a/321val_cls3_m224t224tta0_mix01_wd3',
              # default='stage3a/321val_cls3_m224t224tta0_mix01_wd3_Lr5',
              # default='stage3a/313val_flatM_normT_mix01',
              )
@click.option("--output_dir", type=str,
              default="~/tmp/ddm_dbg",
              )
@click.option("--fold", type=int,
              default=0,
              )
@click.option("--seed", type=int,
              default=1,
              )
@click.option("--gpu", type=str,
              default="0",
              # default="1",
              )
@click.option("--version", type=str, default="v4")
@click.option("--timm_name", type=str,
              # default="hrnet_w64",
              default="resnet34",
              )
@click.option("--debug", type=bool,
              # default=True,
              default=False,
              )
def main(action, exp, fold, seed, gpu, debug, version, timm_name, output_dir):
    set_dbg(debug)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if action == "train":
        print(f'cfg_name={exp}, timm_name={timm_name}, fold={fold}, seed={seed}')
        cfg = common_utils.load_config_data(
            cfg_name=exp, main_cfg=main_cfg,
            timm_name=timm_name, fold=fold, seed=seed, version=version)
        cfg.OUTPUT_DIR = expanduser(output_dir)
        train_and_predict(cfg=cfg)

    if action == "train_all_folds":
        for fold in main_cfg.folds:
            print(f'cfg_name={exp}, timm_name={timm_name}, fold={fold}, seed={seed}')
            cfg = load_config_data(
                cfg_name=exp, main_cfg=main_cfg,
                timm_name=timm_name, fold=fold, seed=seed, version=version)
            cfg.OUTPUT_DIR = expanduser(output_dir)
            train_and_predict(cfg=cfg)


def build_model(cfg, pretrained=True):
    dbg('cfg.model_params')
    pprint(cfg.model_params)
    model_params = common_utils.safe_clone(cfg.model_params)
    name = model_params.model_cls
    cls = models_cls.__dict__[name]
    del model_params['model_cls']
    del model_params['model_type']
    model = cls(cfg=cfg, pretrained=pretrained, **model_params)
    return model


def train_and_predict(*, cfg):
    train(cfg=cfg)
    # NOTE: "val" is "oof" and calculated in train
    # for dataset_type in ['test_val', 'orig_val']:
    for dataset_type in ['test_val']:  # no need to re-calc orig_val
        for best_tag in main_cfg.save_oof_tags:
            predict(cfg=cfg, dataset_type=dataset_type, best_tag=best_tag)


def train(*, cfg):
    seed_everything(cfg.seed)

    # NOTE: cfg == experiment cfg; main_cfg == project cfg
    load_dir_names(cfg)
    logger = SummaryWriter(log_dir=cfg.tensorboard_dir)

    train_params = cfg.train_params
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    data_loaders = my_dataset.make_loaders(cfg)
    model = build_model(cfg)
    print(model.__class__.__name__)
    model = model.cuda()
    model.train()

    initial_lr = float(train_params.initial_lr)
    weight_decay = 1e-2  # todo NOTE default weight_decay = 1e-2 in AdamW
    if hasattr(train_params, 'wd'):
        weight_decay = train_params.wd
    if train_params.optimizer == "adamW":
        # assert weight_decay == 0, 'TODO check'
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif train_params.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif train_params.optimizer == "sgdn":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=True,
                                    weight_decay=weight_decay)
    elif train_params.optimizer == "sgd09":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, nesterov=False,
                                    weight_decay=weight_decay)
    else:
        raise RuntimeError("Invalid optimiser" + train_params.optimizer)

    n_epochs = train_params.n_epochs
    train_loader = data_loaders['train']
    num_train_steps = int(len(train_loader) * n_epochs)
    num_warmup_steps = int(len(train_loader) * train_params.n_warmup_epochs)
    scheduler = get_scheduler(train_params, optimizer, num_train_steps=num_train_steps,
                              num_warmup_steps=num_warmup_steps)

    grad_clip_value = train_params.get("grad_clip", 8.0)
    print(f"grad clip: {grad_clip_value}")

    cr_cls = torch.nn.BCEWithLogitsLoss()
    min_oof_loss = None
    min_oof_neg_acc = None
    train_step_count = 0
    for epoch in tqdm(range(n_epochs)):
        cfg.epoch = epoch
        # for phase in ["train", "val", "orig_val", "test_val"]:
        for phase in ["train", "val"]:  # calc "test_val" separately
            model.train(phase == "train")
            epoch_loss = common_utils.AverageMeter()
            epoch_grad = common_utils.AverageMeter()
            data_loader = data_loaders[phase]
            y_true = []
            y_pred = []
            optimizer.zero_grad()
            data_iter = tqdm(data_loader, disable=False)
            for data in data_iter:
                with torch.set_grad_enabled(phase == "train"):
                    image = data['image'].float().cuda()
                    label = data['label'].float().cuda()
                    if phase == 'train':
                        label = label / (1 + 2 * train_params.labels_smooth) + train_params.labels_smooth
                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                        model_outputs = model(image)
                        loss = cr_cls(model_outputs, label)
                    if phase == "train":
                        train_step_count += 1
                        scaler.scale(loss).backward()
                        # Unscales the gradients of optimizer's assigned params in-place
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                        epoch_grad.update(grad_norm.detach().item(), cfg.batch_size)
                        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                        scaler.step(optimizer)
                        scaler.update()
                        logger.add_scalar("lr_steps", scheduler.get_last_lr()[0], train_step_count)
                        logger.add_scalar("grad_steps", grad_norm.detach().item(), train_step_count)
                        scheduler.step()

                    y_true.append(data['label'].float().numpy())
                    y_pred.append(torch.sigmoid(model_outputs).detach().cpu().numpy())
                    epoch_loss.update(loss.detach().item(), cfg.batch_size)
                    data_iter.set_description(
                        f"{epoch} {phase}"
                        f" Loss {epoch_loss.avg:1.4f}"
                        f" Grad {epoch_grad.avg:1.4f}"
                    )
                    del loss

            y_true = np.concatenate(y_true, axis=0).astype(np.float32)
            y_pred = np.concatenate(y_pred, axis=0).astype(np.float32)
            if phase == "val":
                if np.isnan(y_pred).sum() > 0:
                    print("NAN predictions")
                    continue
                neg_acc = 1 - ((y_true > 0.5) == (y_pred > 0.5)).mean()
                logger.add_scalar("neg_acc", neg_acc, epoch)
                loss_clip4 = loss_with_clip(y_pred=y_pred, y_true=y_true, clip_eps=1e-4)
                loss_clip3 = loss_with_clip(y_pred=y_pred, y_true=y_true, clip_eps=1e-3)
                loss_clip2 = loss_with_clip(y_pred=y_pred, y_true=y_true, clip_eps=1e-2)
                logger.add_scalar("bce_clip_1e-4", loss_clip4, epoch)
                logger.add_scalar("bce_clip_1e-3", loss_clip3, epoch)
                logger.add_scalar("bce_clip_1e-2", loss_clip2, epoch)
                print(
                    f"{epoch} {phase}"
                    f" Loss={epoch_loss.avg:1.4f}"
                    f" 1-acc={neg_acc:1.4f}"
                    f" c4={loss_clip4:1.4f}"
                    f" c3={loss_clip3:1.4f}"
                    f" c2={loss_clip2:1.4f}"
                    f" {cfg.model_str} {cfg.fold}"
                )
                if 'loss_val' in cfg.save_oof_tags:
                    min_oof_loss = save_oof_if_new_min(
                        cfg=cfg, model=model, new_val=epoch_loss.avg,
                        curr_min=min_oof_loss,
                        y_true=y_true, y_pred=y_pred, best_tag='loss_val'
                    )
                if 'oof_neg_acc' in cfg.save_oof_tags:
                    min_oof_neg_acc = save_oof_if_new_min(
                        cfg=cfg, model=model, new_val=neg_acc,
                        curr_min=min_oof_neg_acc,
                        y_true=y_true, y_pred=y_pred, best_tag='oof_neg_acc'
                    )

            logger.add_scalar(f"loss_{phase}", epoch_loss.avg, epoch)
            if phase == "train":
                logger.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
                logger.add_scalar("grad", epoch_grad.avg, epoch)
            logger.flush()
    del model


def loss_with_clip(*, y_pred, y_true, clip_eps):
    loss_clip = F.binary_cross_entropy(
        torch.from_numpy(np.clip(y_pred, clip_eps, 1 - clip_eps)),
        torch.from_numpy(y_true))
    return loss_clip


def load_checkpoint(*, cfg, model, best_tag):
    fpath = f"{cfg.checkpoints_dir}/best_{best_tag}.pt"
    print(f'Trying to load: {fpath}')
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f'loaded OK')


def predict(*,
            cfg,
            dataset_type,  # "val", "orig_val", "test_val"
            best_tag,  # 'val_loss', 'val_neg_acc'
            ):
    seed_everything(cfg.seed)

    load_dir_names(cfg)
    data_loaders = my_dataset.make_loaders(cfg)
    model = build_model(cfg, pretrained=False)
    print(model.__class__.__name__)
    # cfg.best_tag = best_tag  # todo? 'loss_val'
    load_checkpoint(cfg=cfg, model=model, best_tag=best_tag)
    model = model.cuda()
    model.eval()

    # for phase in ["train", "val", "orig_val", "test_val"]:
    phase = dataset_type
    data_loader = data_loaders[phase]
    y_true = []
    y_pred = []
    data_iter = tqdm(data_loader, disable=False)
    saved = set_dbg(False)
    for data in data_iter:
        with torch.set_grad_enabled(False):
            image = data['image'].float().cuda()
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                pred = model(image)
            y_true.append(data['label'].float().numpy())
            y_pred.append(torch.sigmoid(pred).detach().cpu().numpy())

    set_dbg(saved)
    y_true = np.concatenate(y_true, axis=0).astype(np.float32)
    y_pred = np.concatenate(y_pred, axis=0).astype(np.float32)
    if phase == "val":
        save_oof(cfg=cfg, y_true=y_true, y_pred=y_pred, best_tag=best_tag)
    if phase == "orig_val":
        save_orig_val(cfg=cfg, y_true=y_true, y_pred=y_pred, best_tag=best_tag)
    if phase == "test_val":
        save_test_val(cfg=cfg, y_true=y_true, y_pred=y_pred, best_tag=best_tag)


def save_oof_if_new_min(*, cfg, model, new_val, curr_min, y_true, y_pred, best_tag):
    if curr_min is not None and curr_min < new_val:  # ignore: not best
        return curr_min

    # best_tag: why saving: best_val_loss, best_val_neg_acc == 1-accuracy
    out_path = f'{cfg.oof_dir}/best_{best_tag}.npz'
    np.savez(out_path, labels=y_true, predictions=y_pred)
    torch.save(
        {"epoch": cfg.epoch,
         f"best_{best_tag}": new_val,
         "model_state_dict": model.state_dict(),
         },
        f"{cfg.checkpoints_dir}/best_{best_tag}.pt",
    )

    return new_val


def save_oof(*, cfg, y_true, y_pred, best_tag):
    out_path = f'{cfg.oof_dir}/best_{best_tag}.npz'
    np.savez(out_path, labels=y_true, predictions=y_pred)


def save_orig_val(*, cfg, y_true, y_pred, best_tag):
    out_path = f'{cfg.orig_val_dir}/best_{best_tag}.npz'
    np.savez(out_path, labels=y_true, predictions=y_pred)


def save_test_val(*, cfg, y_true, y_pred, best_tag):
    out_path = f'{cfg.test_val_dir}/best_{best_tag}.npz'
    np.savez(out_path, labels=y_true, predictions=y_pred)


if __name__ == "__main__":
    main()
