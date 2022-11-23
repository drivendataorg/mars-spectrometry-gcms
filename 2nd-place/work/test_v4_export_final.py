import glob
import shutil
import pprint
from collections import defaultdict
from os.path import expanduser, dirname, basename
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, LogisticRegressionCV
from config_main_v4 import cfg as main_cfg
import common_utils_v4 as common_utils
from common_utils_v4 import *
from pprint import pformat


@click.command()
@click.option("--action", type=str,
              # default='test_results',
              default='export_models',
              )
@click.option("--results_dir", type=str,
              # !!! NOTE ddm_v4_outputs is my orig submission, but it does not have model weights
              # default='~/tmp/ddm_v4_outputs',  # collected from ~/tmp/ddm_val with 'export_models'

              # default='~/tmp/ddm_rerun221104',  # rerun
              # default="~/tmp/ddm_rerun221105", #/
              # default='~/tmp/ddm_rerun221105b',  # rerun
              # default='~/tmp/ddm_rerun221104_and_rerun221105b',  # rerun
              # default="~/tmp/ddm_rerun221104_05b_06e",

              # default="~/tmp/ddm_rerun221104_05_05b_06e",

              # default="~/tmp/ddm_rerun221106c_log1000",
              # default="~/tmp/ddm_rerun221106d_log1000",
              # default="~/tmp/ddm_rerun221106e_log1000",
              # default="~/tmp/ddm_rerun06c_06d_06e",
              # default="~/tmp/ddm_rerun04_05b_06c_06d_06e",

              # default='~/tmp/ddm_rerun221105b',  # rerun
              default='~/tmp/ddm_rerun221105b_dbg',  # rerun

              )
@click.option("--mix_model", type=str,
              default='mean_logits_clip1e4',  #
              # default='median',
              # default='mean',
              # default='median_logits_clip1e4',  #
              )
@click.option("--n_export_models", type=int, default=30)
@click.option("--version", type=str,
              default="v4",  # the last version used in submissions
              )
@click.option("--debug", type=bool,
              default=True,
              # default=False,
              )
def main(action, debug, version, results_dir,
         mix_model, n_export_models):
    set_dbg(debug)
    main_cfg.n_export_models = n_export_models
    main_cfg.mix_model = mix_model
    main_cfg.results_dir = results_dir

    # main_cfg.export_dir = f'{results_dir}_test'
    main_cfg.export_dir = '~/tmp/ddm_2nd_place_weights'

    main_cfg.version = version

    # if action == "test_results":
    #     test_results()

    # main_cfg.MODELS = [
    #     ('stage3b/302val_cls3_normM', 'resnet34', '1'),
    #     ('stage3b/302val_cls3_normM_crop', 'resnet34', '1'),
    # ]

    # from: test_results/oof0.085937_ddm_rerun04_05b_06c_06d_06e_test_mean_logits_clip1e4.py
    main_cfg.MODELS = [('stage3a/302val_cls3_normM', 'hrnet_w64', '3', 'loss_val'),
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

    dirs = [
        '~/tmp/ddm_rerun221104',  # rerun
        '~/tmp/ddm_rerun221105b',  # rerun
        "~/tmp/ddm_rerun221106c_log1000",
        "~/tmp/ddm_rerun221106d_log1000",
        "~/tmp/ddm_rerun221106e_log1000",
    ]
    for main_cfg.results_dir in dirs:
        # if action == "export_models":
        export_models_by_list(main_cfg.MODELS)


def export_models_by_list(model_info_list):
    main_cfg.results_dir = expanduser(main_cfg.results_dir)
    print(f'results_dir = {main_cfg.results_dir}')
    main_cfg.export_dir = expanduser(main_cfg.export_dir)
    print(f'export_dir = {main_cfg.export_dir}')

    for model_info in model_info_list:
        export_model(model_info=model_info, copy_weights=True)


def test_results():
    main_cfg.results_dir = expanduser(main_cfg.results_dir)
    print(f'results_dir = {main_cfg.results_dir}')
    main_cfg.export_dir = expanduser(main_cfg.export_dir)
    print(f'export_dir = {main_cfg.export_dir}')

    models = {}
    for best_tag in main_cfg.save_oof_tags:
        res = collect_models(best_tag=best_tag)
        models.update(res)

    models = sorted(list(models.keys()))
    main_cfg.check_models = models

    # NOTE main_cfg is global
    load_all_preds()
    check_importance_v3_grow()
    # check_importance_v4_drop()


def collect_models(*, best_tag):
    dir_path = expanduser(main_cfg.results_dir)
    mask = join(dir_path, f'test_val/{main_cfg.version}/*/*/*/best_{best_tag}.npz')  # test_val is calculated last
    flist = sorted(glob.glob(mask))
    res_folds = defaultdict(list)
    for fpath in flist:
        dbg('fpath')
        names = fpath.split('/')
        dbg('names')
        timm_name_fold_seed = names[-2]
        dbg('timm_name_fold_seed')

        fold_seed_len = len('_fold0_seed1')
        timm_name = timm_name_fold_seed[:-fold_seed_len]
        fold_seed = timm_name_fold_seed[-fold_seed_len + 1:]
        fold, seed = fold_seed.split('_')
        fold = int(fold[-1])
        seed = int(seed[-1])

        cfg_name = f'{names[-4]}/{names[-3]}'  # NOTE it has stage#
        dbg('cfg_name')
        key = (cfg_name, timm_name, seed, best_tag)
        res_folds[key] += [fold]

    print(f'FOUND {len(res_folds)} models')
    res_folds = filter_valid_folds(res_folds)
    print(f'FOUND {len(res_folds)} models WITH ALL FOLDS')
    return res_folds


def filter_valid_folds(res_folds):
    ret = {}
    for k, folds in res_folds.items():
        folds = sorted(folds)
        if folds == main_cfg.folds:
            dbg('k')
            ret[k] = folds
    return ret


def check_importance(predictions, labels):
    mean_pred = torch.stack([p[1] for p in predictions], dim=0).mean(dim=0)

    l = float(F.binary_cross_entropy(mean_pred, labels))
    print(f'All models {l:0.4f} {len(predictions)}')

    while len(predictions) > 1:
        l_without_model = [
            float(F.binary_cross_entropy(
                torch.stack([p[1] for i, p in enumerate(predictions) if i != n], dim=0).mean(dim=0), labels))
            for n in range(len(predictions))
        ]
        weak_model = np.argmin(l_without_model)
        l_next = l_without_model[weak_model]
        print(
            f'{l_next:0.4f} {l_next - l:0.4f} {predictions[weak_model][0]:32s}  {predictions[weak_model][2]:0.3f}  {len(predictions) - 1}')

        l = l_next
        del predictions[weak_model]

    print(predictions[0][0])


class MeanModel:
    def fit(self, X, y, sample_weight=None):
        pass

    def predict_proba(self, x):
        model_axis = 1
        pos = np.mean(x, axis=model_axis, keepdims=False)
        # pos = np.mean(x, axis=model_axis, keepdims=True)
        # neg = 1 - pos
        # ret = np.concatenate([neg, pos], axis=model_axis)
        # return ret
        return pos


def inverse_sigmoid(x, eps=1e-8):
    x = torch.clip(x, eps, 1 - eps)
    x = torch.log(x / (1 - x))
    return x


class LogitsModel:
    def __init__(self, clip_limit, how='median'):
        self.clip_limit = clip_limit
        self.how = how

    def fit(self, X, y, sample_weight=None):
        pass

    def predict_proba(self, x):
        dbg(x)
        x = np.clip(x, self.clip_limit, 1 - self.clip_limit)
        dbg(x)

        x = torch.Tensor(x)
        dbg(x)
        x1 = inverse_sigmoid(x)
        dbg(x1)
        x2 = torch.sigmoid(x1)
        dbg(x2)
        assert torch.all((x2 - x).abs() < 1e-5)

        x = inverse_sigmoid(x)
        dbg(x)
        model_axis = 1
        if self.how == 'median':
            x, _ = torch.median(x, dim=model_axis, keepdim=False)
        elif self.how == 'mean':
            x = torch.mean(x, dim=model_axis, keepdim=False)
        dbg(x)
        x = torch.sigmoid(x)
        dbg(x)
        x = x.detach().cpu().numpy()
        dbg(x)
        return x


class MedianModel:
    def fit(self, X, y, sample_weight=None):
        pass

    def predict_proba(self, x):
        model_axis = 1
        dbg(x)
        pos = np.median(x, axis=model_axis, keepdims=False)
        dbg(pos)
        return pos


def prep_for_ml(preds):
    dbg(preds)  # models, samples, targets
    preds = np.moveaxis(preds, 0, 1)
    dbg(preds)  # samples, models, targets
    # preds = preds.reshape((preds.shape[0], -1))
    # dbg(preds)  # samples, models * targets
    return preds


def search_best_ml(*, oof_preds, oof_labels,
                   verb, test_preds):
    dbg(oof_preds)  # models, samples, targets
    oof_x = prep_for_ml(oof_preds)
    test_x = prep_for_ml(test_preds)
    dbg(oof_x)

    oof_y = oof_labels
    dbg(oof_y)
    # grid search:
    best_oof_loss = None
    best_oof_clip = None
    best_oof_ml = None
    if main_cfg.mix_model == 'mean':
        ml = MeanModel()
    elif main_cfg.mix_model == 'median':
        ml = MedianModel()
    elif main_cfg.mix_model == 'mean_logits_clip1e4':
        ml = LogitsModel(clip_limit=1e-4, how='mean')
    elif main_cfg.mix_model == 'mean_logits_clip1e5':
        ml = LogitsModel(clip_limit=1e-5, how='mean')
    elif main_cfg.mix_model == 'median_logits_clip1e4':
        ml = LogitsModel(clip_limit=1e-4)
    elif main_cfg.mix_model == 'median_logits_clip1e5':
        ml = LogitsModel(clip_limit=1e-5)
    elif main_cfg.mix_model == 'median_logits_clip1e3':
        ml = LogitsModel(clip_limit=1e-3)
    elif main_cfg.mix_model == 'median_logits_clip1e2':
        ml = LogitsModel(clip_limit=1e-2)
    else:
        print(f'ERROR: main_cfg.mix_model = {main_cfg.mix_model}')
        exit()
    ml.fit(oof_x, oof_y)

    # TEST:
    # test_probs = ml.predict_proba(test_x)[:, -1]
    test_probs = ml.predict_proba(test_x).astype(test_preds.dtype)
    # test_probs = test_probs.reshape(test_preds.shape[1:]).astype(test_preds.dtype)
    # OOF:
    # yp = ml.predict_proba(oof_x)[:, -1]
    yp = ml.predict_proba(oof_x).astype(oof_preds.dtype)
    # yp = yp.reshape(oof_labels.shape).astype(oof_preds.dtype)
    # todo: return clip
    loss, clip = common_utils.print_metrics(labels=oof_labels, preds=yp, verb=False)
    if best_oof_loss is None or best_oof_loss > loss:
        best_oof_loss = loss
        best_oof_loss_test_probs = test_probs
        best_oof_clip = clip
        best_oof_ml = ml

    ret = dict(
        best_oof_loss=best_oof_loss,
        best_oof_loss_test_probs=best_oof_loss_test_probs,
        best_oof_model=best_oof_ml, best_oof_clip=best_oof_clip,
    )
    if verb:
        pprint(ret)
    return ret


def export_model(model_info, copy_weights):
    best_tag_str = None
    if len(list(model_info)) == 4:
        cfg_name, timm_name, seed, best_tag = model_info
        best_tag_str = f'best_{best_tag}.'
    else:
        cfg_name, timm_name, seed = model_info
    seed_str = f'seed{seed}'
    timm_str = f'/{timm_name}_fold'
    in_dir = expanduser(main_cfg.results_dir)
    out_dir = expanduser(main_cfg.export_dir)
    for root, dirs, files in tqdm(os.walk(in_dir)):
        for f in files:
            fpath = os.path.join(root, f)
            dbg('fpath')
            if not (cfg_name in fpath and timm_str in fpath and seed_str in fpath):
                continue
            # if not (best_tag_str in fpath):
            #     continue
            ext = fpath.split('.')[-1]
            if not copy_weights and ext == 'pt':
                continue  # do not copy weights: too slow/large
            out_path = fpath.replace(in_dir, out_dir)
            os.makedirs(dirname(out_path), exist_ok=True)
            dbg('fpath')
            dbg('out_path')
            shutil.copyfile(src=fpath, dst=out_path)


def export_test_preds(best_try_list=None):
    if best_try_list is not None:
        oof_preds = main_cfg.oof_preds[best_try_list]
        test_preds = main_cfg.test_preds[best_try_list]
    else:
        oof_preds = main_cfg.oof_preds
        test_preds = main_cfg.test_preds
    oof_labels = main_cfg.oof_labels
    dbg(oof_preds)
    res = search_best_ml(
        oof_preds=oof_preds, oof_labels=oof_labels,
        test_preds=test_preds,
        verb=False)

    test_probs = res["best_oof_loss_test_probs"]  # could be training on orig_val
    best_oof_loss = res["best_oof_loss"]  # could be training on orig_val

    fpath = join(main_cfg.DATA_DIR, 'metadata.csv')
    metadata = pd.read_csv(fpath)
    metadata.fillna(0, inplace=True)
    sample_ids = list(metadata[metadata.split != 'train'].sample_id)
    df = pd.DataFrame({"sample_id": sample_ids})
    for idx, col in enumerate(main_cfg.cls_labels):
        probs = test_probs[:, idx]
        df[col] = probs
    dbg(df)
    export_tag = basename(main_cfg.export_dir)

    best_models = [tuple(main_cfg.check_models[i]) for i in best_try_list]

    # fname = f'{export_tag}_oof{best_oof_loss:0.5f}'
    fname = f'oof{best_oof_loss:0.6f}_{export_tag}'
    fname = f'{fname}_{main_cfg.mix_model}.csv'
    out_path = f'test_results/{fname}'
    os.makedirs(dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print_save_best_models(out_path, best_models)

    out_path = join(main_cfg.export_dir, fname)
    os.makedirs(dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print_save_best_models(out_path, best_models)


def print_save_best_models(out_path, best_models):
    out_path = out_path.replace('.csv', '.py')
    with open(out_path, "w") as fout:
        text = pformat(best_models)
        print('BEST models:')
        print(text)
        text = f'cfg.MODELS = {text}'
        fout.write(text)


def check_importance_v3_grow():
    n_export_models = main_cfg.n_export_models
    oof_preds = main_cfg.oof_preds
    oof_labels = main_cfg.oof_labels
    test_preds = main_cfg.test_preds
    dbg(oof_preds)
    res = search_best_ml(
        oof_preds=oof_preds, oof_labels=oof_labels,
        test_preds=test_preds,
        verb=True)
    best_try_list = []
    best_oof_loss = 1
    check_models = main_cfg.check_models
    while len(best_try_list) < n_export_models and len(best_try_list) < len(check_models):
        oof_list, idx_list = [], []
        oof_clip_list = []
        for n in range(len(oof_preds)):
            if n in best_try_list:
                continue
            try_list = best_try_list + [n]
            try_preds = oof_preds[try_list]
            try_test_preds = test_preds[try_list]
            try_res = search_best_ml(
                oof_preds=try_preds, oof_labels=oof_labels,
                test_preds=try_test_preds,
                verb=False)
            oof_loss = try_res['best_oof_loss']
            oof_clip = try_res['best_oof_clip']
            oof_list += [oof_loss]
            oof_clip_list += [oof_clip]
            idx_list += [n]
            if len(best_try_list) == 0:  # print all
                print(f'oof_loss={oof_loss:0.4f}, {check_models[n]}')

        tmp_idx = np.argmin(oof_list)

        try_oof_loss = oof_list[tmp_idx]
        try_oof_clip = oof_clip_list[tmp_idx]
        best_oof_idx = idx_list[tmp_idx]  #

        best_try_list += [best_oof_idx]  # <----
        best_new_model = check_models[best_oof_idx]

        print(
            f'{check_models[best_oof_idx]} n_oof_models={len(best_try_list)}')
        print(
            f'best_oof_loss={best_oof_loss:0.4f}, {try_oof_loss - best_oof_loss:0.4f}, '
            f'last={try_oof_loss:0.4f}   clip={try_oof_clip:0.4f}')

        export_model(best_new_model, copy_weights=False)  # <---------- NOTE: with medians "not better" is NOT LAST

        if best_oof_loss is not None and best_oof_loss < try_oof_loss:
            print('NOT BETTER!!!!!!!!!!!!!!!!')
            continue
        if best_oof_loss is None or best_oof_loss > try_oof_loss:
            best_oof_loss = try_oof_loss
            export_test_preds(best_try_list)  # recalc
            # print_models(check_models, best_try_list)


def check_importance_v4_drop():
    n_export_models = main_cfg.n_export_models
    oof_preds = main_cfg.oof_preds
    oof_labels = main_cfg.oof_labels
    test_preds = main_cfg.test_preds
    dbg(oof_preds)
    all_res = search_best_ml(
        oof_preds=oof_preds, oof_labels=oof_labels,
        test_preds=test_preds,
        verb=True)

    best_try_list = list(np.arange(len(main_cfg.check_models)))
    best_oof_loss = all_res['best_oof_loss']
    check_models = main_cfg.check_models

    while len(best_try_list) > 1:
        oof_list, idx_list = [], []
        oof_clip_list = []
        for n in range(len(oof_preds)):
            if n not in best_try_list:
                continue
            # try_list = best_try_list + [n]
            try_list = best_try_list.copy()
            try_list.remove(n)
            try_preds = oof_preds[try_list]
            try_test_preds = test_preds[try_list]
            try_res = search_best_ml(
                oof_preds=try_preds, oof_labels=oof_labels,
                test_preds=try_test_preds,
                verb=False)
            oof_loss = try_res['best_oof_loss']
            oof_clip = try_res['best_oof_clip']
            oof_list += [oof_loss]
            oof_clip_list += [oof_clip]
            idx_list += [n]

        # remember oof_list is new each time
        tmp_idx = np.argmin(oof_list)
        try_oof_loss = oof_list[tmp_idx]
        try_oof_clip = oof_clip_list[tmp_idx]
        del_oof_idx = idx_list[tmp_idx]  #
        best_try_list.remove(del_oof_idx)  # <----

        print(
            f'REMOVED {check_models[del_oof_idx]} n_oof_models={len(best_try_list)}')
        print(
            f'best_oof_loss={best_oof_loss:0.4f}, {try_oof_loss - best_oof_loss:0.4f}, '
            f'last={try_oof_loss:0.4f}   clip={try_oof_clip:0.4f}')

        if best_oof_loss is not None and best_oof_loss < try_oof_loss:
            print('NOT BETTER!!!!!!!!!!!!!!!!')
            continue
        if best_oof_loss is None or best_oof_loss > try_oof_loss:
            best_oof_loss = try_oof_loss
            export_test_preds(best_try_list)  # recalc


def load_all_preds():
    oof_preds = []
    oof_labels = np.array([])
    test_labels, val_labels = None, None  # test is test+val
    test_preds, val_preds = [], []
    keep_models = []
    for info in main_cfg.check_models:
        cfg_name, timm_name, seed, best_tag = info
        model_labels = []
        model_oof_preds = []
        model_test_preds = []
        try:
            for fold in main_cfg.folds:
                cfg = common_utils.load_config_data(
                    cfg_name=cfg_name,
                    main_cfg=main_cfg,
                    timm_name=timm_name, fold=fold,
                    seed=seed, version=main_cfg.version)
                cfg.OUTPUT_DIR = main_cfg.results_dir  # todo NOTE redirecting here
                load_dir_names(cfg)
                data = np.load(f'{cfg.oof_dir}/best_{best_tag}.npz')
                model_labels += [data['labels'].copy()]
                model_oof_preds += [data['predictions'].copy()]

                data = np.load(f'{cfg.test_val_dir}/best_{best_tag}.npz')
                if test_labels is None:  # all vals should be the same
                    test_labels = data['labels'].copy()
                else:
                    assert np.all(test_labels == data['labels'])
                model_test_preds += [data['predictions'].copy()]
        except Exception as e:
            print(e)
            continue

        oof_labels = np.concatenate(model_labels, axis=0)
        model_oof_preds = np.concatenate(model_oof_preds, axis=0)
        model_test_preds = np.mean(model_test_preds, axis=0)  # note mean
        if len(model_oof_preds) != 809:  # todo: temp, bug in train_val setup
            continue
        if np.sum(np.isnan(model_oof_preds)) > 0:
            continue
        if np.sum(np.isnan(model_test_preds)) > 0:
            continue
        keep_models += [info]
        oof_preds += [model_oof_preds]
        test_preds += [model_test_preds]

    main_cfg.check_models = np.array(keep_models)
    main_cfg.oof_labels = np.array(oof_labels)
    main_cfg.oof_preds = np.array(oof_preds)
    main_cfg.val_preds = np.array(val_preds)
    main_cfg.test_preds = np.array(test_preds)  # todo
    dbg(main_cfg.oof_preds)


if __name__ == "__main__":
    main()
