import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt

import glob
import sys
import os.path
from os.path import join, isfile, expanduser, dirname
import cv2
import joblib
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A

import common_utils_v4 as common_utils
from common_utils_v4 import *

import warnings

warnings.filterwarnings("ignore")

# # SORT_SAVE = True
# SORT_SAVE = False
# # SHOW = True
# SHOW = False

def make_loaders(cfg):
    from torch.utils.data import DataLoader

    dataset_train = MarsSpectrometryDataset(
        cfg=cfg, is_training=True, output_pytorch_tensor=True, fold=cfg.fold,
        **cfg.dataset_params)
    dataset_val = MarsSpectrometryDataset(
        cfg=cfg, is_training=False, output_pytorch_tensor=True, fold=cfg.fold,
        **cfg.dataset_params)

    if 'dataset_type' in cfg.dataset_params:
        # todo: fix for default dataset_type='train'
        cfg.dataset_params.pop('dataset_type')

    dataset_orig_val = MarsSpectrometryDataset(
        dataset_type='val',  # <--- orig_val
        cfg=cfg, is_training=False, output_pytorch_tensor=True, fold=cfg.fold,
        **cfg.dataset_params)

    dataset_test_val = MarsSpectrometryDataset(
        dataset_type='test_val',  # <--- test + orig_val
        cfg=cfg, is_training=False, output_pytorch_tensor=True, fold=cfg.fold,
        **cfg.dataset_params)

    batch_size = cfg.train_data_loader.batch_size
    cfg.batch_size = batch_size
    num_workers = 0 if get_dbg() else cfg.train_data_loader.num_workers  # fixed
    # num_workers = 0 if get_dbg() else multiprocessing.cpu_count()  # !!!different comps may have diff count!!!
    # https://pytorch.org/docs/stable/notes/randomness.html
    # def seed_worker(worker_id):
    #     worker_seed = torch.initial_seed() % 2**32
    #     numpy.random.seed(worker_seed)
    #     random.seed(worker_seed)
    #
    # g = torch.Generator()
    # g.manual_seed(0)
    #
    # DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     worker_init_fn=seed_worker,
    #     generator=g,
    # )
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    data_loaders = {
        "train": DataLoader(
            dataset_train,
            num_workers=num_workers,
            worker_init_fn=common_utils.seed_worker,
            shuffle=True, batch_size=batch_size,
            drop_last=True, pin_memory=True,
            generator=g,
        ),
        "val": DataLoader(dataset_val, num_workers=num_workers,
                          shuffle=False, batch_size=batch_size),
        "orig_val": DataLoader(dataset_orig_val, num_workers=num_workers,
                               shuffle=False, batch_size=batch_size),
        "test_val": DataLoader(dataset_test_val, num_workers=num_workers,
                               shuffle=False, batch_size=batch_size),
    }
    return data_loaders


def get_train_folds(*, cfg, is_training, fold):
    folds = pd.read_csv(cfg.folds_csv_path)
    dbg(folds)
    fold = int(fold)
    if is_training:
        folds = folds[folds.fold != fold]
    else:
        folds = folds[folds.fold == fold]
    dbg(folds)
    sample_ids = list(folds.sample_id.values)
    labels = pd.read_csv(cfg.train_labels_path, index_col='sample_id')
    return sample_ids, labels


def norm_min0_max1(v):
    v = v - v.min()
    v = v / v.max()
    return v


def show_sample(cfg, x, tag):
    if not cfg.SHOW_SAMPLES:
        return
    if len(x.shape) == 3:
        x = np.sum(x, axis=-1, keepdims=True)
        x = np.repeat(x, repeats=3, axis=-1)
    else:
        x = np.repeat(x[:, :, None], repeats=3, axis=-1)

    nrows, ncols = x.shape[:2]
    xrows = np.arange(nrows) / (nrows-1)
    xcols = np.arange(ncols) / (ncols-1)
    xcols = np.repeat(xcols[None, :], nrows, axis=0)
    xrows = np.repeat(xrows[:, None], ncols, axis=1)

    x = norm_min0_max1(x)
    x[:, :, 0] = xrows/2  # /2 to make it nicer
    x[:, :, 1] = xcols/2
    x = np.flipud(x)  # flip mass: mass=0 will be at the bottom

    z = (x * 255).astype(np.uint8)

    # z = ((1-x) * 255).astype(np.uint8)
    fpath = expanduser(f'~/tmp/ddm/dbg/{tag}.png')
    os.makedirs(dirname(fpath), exist_ok=True)
    cv2.imwrite(fpath, z)
    plt.imshow(z)
    plt.xlabel(cfg.col_t)
    plt.ylabel(cfg.col_m)
    plt.show()

def convert_yaxis_to_ampl(x):
    m_bins, t_bins = x.shape[:2]
    idx = np.array(list(np.ndindex(x.shape)))
    x = x / x.max() * (m_bins-1)
    x = np.round(x).astype(int)
    dbg(x)
    new_rows = int(np.max(x) + 1)
    assert new_rows == m_bins
    new_cols = x.shape[1]
    old_rows = x.shape[0]
    old_rgb = x.shape[2]
    # z = np.zeros((new_rows, new_cols, 3))
    z = np.zeros((new_rows, new_cols, old_rgb))
    for r, c, rgb in idx:
        a = x[r, c, rgb]
        new_r = (r * (new_rows-1) / (old_rows-1))
        z[a, c, rgb] = new_r
        z[a, c, 2-rgb] = new_rows - new_r
        # z[a, c, 0] = new_r
        # z[a, c, 1] = new_rows - new_r
    z = np.flipud(z).copy()
    dbg(z)
    # if SHOW:
    #     plt.imshow(z.astype(np.uint8))
    #     plt.xlabel(cfg.col_t)
    #     plt.ylabel(cfg.col_a)
    #     plt.show()
    return z

def sort_save_sample(*, cfg, meta, x, data_tag, sample_dict):
    #         sample_dict = {
    #             'item': idx,
    #             'image': x,
    #             'sample_id': sample_id,
    #             'sample_id2': sample_id2,
    #             'label': label_values,
    #             'label2': label_values2,
    #             'derivatized': str(metadata['derivatized'])
    #             # 'instrument_type': metadata['instrument_type']
    #         }
    if not cfg.SORT_SAVE_SAMPLES:
        return

    # label_values = [labels[k] for k in self.cfg.cls_labels]
    sample_id = sample_dict['sample_id']
    label_values = sample_dict['label']
    y = np.array(label_values).astype(bool)
    cls_labels = np.array(cfg.cls_labels)
    names = cls_labels[y]

    y_code = "".join(list(y.astype(int).astype(str)))
    dir_name = f'{"_".join(names)}'
    dir_name = f'{y_code}_{dir_name}'
    fpath = expanduser(f'~/tmp/ddm_dbg/{dir_name}/{sample_id}_{data_tag}.png')
    os.makedirs(dirname(fpath), exist_ok=True)
    dbg('fpath')

    x = norm_min0_max1(x)
    x = (x * 255).astype(np.uint8)
    # if x.shape[0] < 512:
    #     new_shape = np.array(x.shape[:2]) * 2
    #     new_shape = list(new_shape.astype(int))
    #     x = cv2.resize(x, new_shape)
    os.makedirs(dirname(fpath), exist_ok=True)
    cv2.imwrite(fpath, x)
    # plt.imshow(z)
    # plt.xlabel(cfg.col_t)
    # plt.ylabel(cfg.col_m)
    # plt.show()



def render_image(*, cfg, df, t_bins, m_bins, max_m):
    dbg(df)
    col_t = cfg.col_t  # time or temp
    col_m = cfg.col_m  # m/z or mass
    col_a = cfg.col_a  # abundance or intensity

    tv = norm_min0_max1(df[col_t].values)
    tv = np.round(tv * (t_bins - 1)).astype(int)
    df[col_t] = tv
    df[col_t] = df[col_t].astype(int)
    dbg(df)
    assert m_bins == max_m + 1, 'CHECK m_bins, max_m'
    df = df.groupby([col_m, col_t])[col_a].sum().reset_index()
    dbg(df)
    mv = df[col_m].values
    tv = df[col_t].values
    av = df[col_a].values
    x = np.zeros((m_bins, t_bins))
    x[mv, tv] = av
    x = x / x.max()
    dbg(x)
    show_sample(cfg=cfg, x=x, tag='dbg_1_render_image')
    return x


def filter_labels(df, pseudos_filter):
    dbg(df)
    x = df.values
    dbg(x)
    ok = (x <= pseudos_filter) | (x >= 1 - pseudos_filter)
    keep = np.all(ok, axis=-1)
    df = df.loc[keep]
    dbg(df)

    x = df.values
    x[x <= pseudos_filter] = 0
    x[x >= 1 - pseudos_filter] = 1
    df[df.columns] = x
    dbg(df)
    return df


def sum_by_time(x):
    m_bins, t_bins = x.shape[:2]
    old_rgb = x.shape[2]
    x = np.sum(x, axis=1)
    idx = np.array(list(np.ndindex(x.shape)))
    x = x / x.max() * (m_bins - 1)
    x = np.round(x).astype(int)
    dbg(x)
    new_rows = int(np.max(x) + 1)
    assert new_rows == m_bins
    z = np.zeros((m_bins, m_bins, old_rgb))
    for r, rgb in idx:
        a = x[r, rgb]
        z[r, 0:a, rgb] = 1

    # low-m TOP-X-axis
    z = np.moveaxis(z, source=0, destination=1)  #
    dbg(z)
    z = np.flipud(z)  #
    dbg(z)
    # if SHOW:
    #     plt.imshow(z.astype(np.uint8))
    #     plt.xlabel(cfg.col_t)
    #     plt.ylabel(cfg.col_a)
    #     plt.show()
    return z


def drop_cols(x, aug_n_drops):
    idx = np.arange(x.shape[1])  # cols
    idx = np.random.choice(idx, size=aug_n_drops, replace=False)
    x[:, idx, :] = 0
    return x


class MarsSpectrometryDataset(Dataset):
    def __init__(self, *,
                 cfg,
                 fold, is_training,
                 m_to_int_mode='round',
                 m_bins=256,
                 t_bins=256,
                 dataset_type='train',
                 output_pytorch_tensor=True,
                 min_clip=1e-5,
                 remove_he=True,
                 min_m=0,
                 max_m=512 - 1,
                 norm_to_one=True,
                 mix_aug_prob=0.0,
                 aug_drops_prob=0.0,
                 aug_n_drops=0,
                 normalize_m_separately=False,
                 normalize_t_separately=False,
                 log_space=True,
                 scale_before_log=-1,
                 mix_type='orig',
                 ):
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.is_training = is_training
        self.t_bins = t_bins
        self.m_bins = m_bins
        self.remove_he = remove_he
        self.max_m = max_m
        self.min_m = min_m
        self.m_to_int_mode = m_to_int_mode
        self.output_pytorch_tensor = output_pytorch_tensor
        self.is_training = is_training
        self.fold = fold
        self.min_clip = min_clip
        self.log_space = log_space
        self.scale_before_log = scale_before_log
        self.norm_to_one = norm_to_one
        self.aug_drops_prob = aug_drops_prob
        self.aug_n_drops = aug_n_drops
        self.mix_aug_prob = mix_aug_prob
        self.mix_type = mix_type
        self.normalize_m_separately = normalize_m_separately
        self.normalize_t_separately = normalize_t_separately
        if not is_training:
            self.mix_aug_prob = 0
            self.aug_drops_prob = 0

        self.sample_ids = []
        fpath = join(cfg.DATA_DIR, 'metadata.csv')
        self.meta = pd.read_csv(fpath, index_col='sample_id')
        self.meta.fillna(0, inplace=True)

        if dataset_type == 'train':
            self.sample_ids, self.labels = get_train_folds(cfg=cfg, is_training=is_training, fold=fold)
        elif dataset_type == 'val':
            self.sample_ids = list(self.meta[self.meta.split == 'val'].index.values)
            self.labels = pd.read_csv(cfg.val_labels_path, index_col='sample_id')
        elif dataset_type == 'train_val':
            self.sample_ids, self.labels = get_train_folds(cfg=cfg, is_training=is_training, fold=fold)
            if is_training:
                val_sample_ids = list(self.meta[self.meta.split == 'val'].index.values)
                self.sample_ids += val_sample_ids
                val_labels = pd.read_csv(cfg.val_labels_path, index_col='sample_id')
                self.labels = pd.concat([self.labels, val_labels], axis=0)
        elif dataset_type == 'test_val':
            self.sample_ids = list(self.meta[self.meta.split != 'train'].index.values)
            self.labels = pd.DataFrame(index=self.sample_ids)
            for col in cfg.cls_labels:
                self.labels[col] = 0.0

        dbg(self.meta)
        self.meta = self.meta.loc[self.sample_ids].copy()
        # sample_id                                     S0807
        # split                                         train
        # derivatized                                     0.0
        # features_path              train_features/S0807.csv
        # features_md5_hash  29fe0e94a96e6e206232248d22e48398
        dbg(self.meta)
        loaded_ids = []
        fpath_list = []
        for sample_id, row in self.meta.iterrows():
            fpath = row['features_path']
            fpath = join(cfg.DATA_DIR, fpath)
            assert isfile(fpath), f'Missing {fpath}'
            fpath_list += [fpath]
            loaded_ids += [sample_id]
        self.sample_ids = loaded_ids
        self.meta['fpath'] = fpath_list
        print(f'dataset_type={dataset_type}, Training={is_training}, num_samples={len(self.sample_ids)} ')

    def __len__(self):
        return len(self.sample_ids)

    def filter_by_m(self, df):
        col = self.cfg.col_m
        if self.m_to_int_mode == 'round':
            mv = df[col].values
            mv = np.round(mv).astype(int)
            dbg(df)
            df[col] = mv
            df[col] = df[col].astype(int)
            dbg(df)
        keep = df[col] <= self.max_m
        keep = keep & (df[col] >= self.min_m)
        if self.remove_he:
            keep = keep & (df[col] != 4)
        dbg(df)
        df = df[keep].reset_index(drop=True)
        dbg(df)
        return df

    def _render_image(self, df):

        x = render_image(
            cfg=self.cfg, df=df,
            t_bins=self.t_bins,
            m_bins=self.m_bins,
            max_m=self.max_m)
        return x

    def save_samples_for_plots(self, df, sample_id):
        if not self.cfg.EXPORT_MAX_MASS_1D:
            return

        df = df.copy()

        # img = self._render_image(df.copy())

        col_t = self.cfg.col_t  # time or temp
        col_m = self.cfg.col_m  # m/z or mass
        col_a = self.cfg.col_a  # abundance or intensity
        mv = np.round(df[col_m].values).astype(int)
        x = df[col_t].values
        y = df[col_a].values

        # select mass with max
        idx = np.argmax(y)
        m = mv[idx]
        df[col_m] = mv
        df[col_m] = df[col_m].astype(int)
        keep = df[col_m] == m
        dbg(df)
        df = df[keep].reset_index(drop=True)
        dbg(df)
        x = df[col_t].values
        y = df[col_a].values

        x = x / x.max()
        assert self.scale_before_log > 99
        y = y / np.max(y) * self.scale_before_log
        y = np.log10(y + 1)
        y = y / np.max(y)
        df[col_a] = y

        plt.scatter(x, y, c='b', s=2, alpha=0.5, label='raw values')

        # tv = norm_min0_max1(df[col_t].values)
        dbg(x)
        x = np.round(x * (self.t_bins - 1)).astype(int)
        dbg(x)
        df[col_t] = x
        df[col_t] = df[col_t].astype(int)
        dbg(df)
        # df = df.groupby([col_m, col_t])[col_a].sum().reset_index()
        df = df.groupby([col_m, col_t])[col_a].mean().reset_index()
        dbg(df)
        mv = df[col_m].values
        x = df[col_t].values
        x = x / x.max()
        y = df[col_a].values

        plt.plot(x, y, c='r', linewidth=2, alpha=0.5, label='192 bins')
        plt.title(f'Sample = {sample_id}, m/z = {m}')
        plt.xlabel(self.cfg.col_t)
        plt.ylabel(self.cfg.col_a)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.legend(loc='upper left')
        plt.legend(loc='upper right')

        fpath = expanduser(f'~/tmp/ddm/dbg/max_mass_1d_plots/{sample_id}.png')
        os.makedirs(dirname(fpath), exist_ok=True)
        plt.savefig(fpath)

        plt.show()
        # if cfg.EXPORT_MAX_MASS_1D:

    def render_item(self, sample_id):
        metadata = self.meta.loc[sample_id]
        # split: train
        # derivatized: nan
        # features_path: train_features/S0045.csv
        # features_md5_hash: ff723860b7a37d31fa1048cf27a051f1
        dbg(metadata)
        fpath = metadata['fpath']
        print('r', end="")
        df = pd.read_csv(fpath)  # was read_sample(sample_id)
        df.fillna(0, inplace=True)
        dbg(df)
        df = self.filter_by_m(df)

        self.save_samples_for_plots(df=df, sample_id=sample_id)

        x = self._render_image(df)

        derivatized = int(metadata['derivatized'])
        zero_idx = 1 - derivatized
        x = np.repeat(x[:, :, None], 3, axis=-1)
        x[:, :, zero_idx] = 0  # encode derivatized as a channel
        x[:, :, 2] = 0
        show_sample(cfg=self.cfg, x=x, tag='dbg_2_render_item')
        x = x.astype(np.float32)
        return x

    def make_data_tag(self):
        tag = f'm_bins{self.m_bins}_t_bins{self.t_bins}_m_to_int_mode{self.m_to_int_mode}_aug0'
        return tag

    def make_img_tag(self):
        img_tag = f'clip{self.min_clip}'
        if self.normalize_m_separately:
            img_tag = f'{img_tag}_normM{int(self.normalize_m_separately)}'
        if self.normalize_t_separately:
            img_tag = f'{img_tag}_normT{int(self.normalize_t_separately)}'
        if self.scale_before_log > 0:
            img_tag = f'{img_tag}_scale_before_log{self.scale_before_log}'
        return img_tag

    def get_item(self, idx):
        sample_id = self.sample_ids[idx]
        labels = self.labels.loc[sample_id]
        label_values = [labels[k] for k in self.cfg.cls_labels]
        label_values = np.array(label_values, dtype=np.float32)
        metadata = self.meta.loc[sample_id]
        # derivatized = int(metadata['derivatized'])
        # split: train
        # derivatized: nan
        # features_path: train_features/S0045.csv
        # features_md5_hash: ff723860b7a37d31fa1048cf27a051f1
        dbg(metadata)
        fpath = metadata['fpath']
        joblib_path = f'{fpath}.joblib'
        tag = self.make_data_tag()
        joblib_dir = join(self.cfg.TMP_DIR, tag)
        # joblib_path = joblib_path.replace(self.cfg.DATA_DIR, expanduser(f'~/tmp/ddm/{tag}'))
        joblib_path = joblib_path.replace(self.cfg.DATA_DIR, joblib_dir)
        if isfile(joblib_path):
            # print('-', end="")
            x = joblib.load(joblib_path)
        else:
            print('+', end="")
            x = self.render_item(sample_id)
            os.makedirs(dirname(joblib_path), exist_ok=True)
            joblib.dump(x, joblib_path)

        if self.normalize_t_separately or self.normalize_m_separately:
            assert self.normalize_t_separately != self.normalize_m_separately
            # p[:, :] /= np.clip(np.max(p, axis=1, keepdims=True), 1e-5, 1e5) + 5e-4
            axis = int(self.normalize_m_separately)
            m_max_arr = np.max(x, axis=axis, keepdims=True)
            dbg(m_max_arr)
            m_max_arr = np.clip(m_max_arr, self.min_clip, None)
            dbg(m_max_arr)
            x = x / m_max_arr

        if self.aug_drops_prob > 0 and np.random.rand() < self.aug_drops_prob:
            x = drop_cols(x, self.aug_n_drops)

        return x, label_values, sample_id, metadata

    def __getitem__(self, idx):
        x, label_values, sample_id, metadata = self.get_item(idx)

        sample_id2 = ''
        label_values2 = np.zeros_like(label_values)
        # assert self.mix_aug_prob == 0, 'check mix_aug_prob'
        if self.mix_aug_prob > 0 and np.random.rand() < self.mix_aug_prob:
            idx2 = np.random.randint(len(self.sample_ids))
            x2, label_values2, sample_id2, metadata2 = self.get_item(idx2)
            if self.mix_type == 'mixup_sum_renorm1':
                x = x.copy() + x2.copy()
                x = x / x.max()  # 0-1
            elif self.mix_type == 'mixup_04_06':
                frac = np.random.uniform(low=0.4, high=0.6)  # to config?
                x = frac * x.copy() + (1 - frac) * x2.copy()
            else:
                if np.random.rand() > 0.5:
                    x = np.maximum(x.copy(), x2.copy())
                else:
                    frac = np.random.uniform(low=0.4, high=0.6)  # to config?
                    # p = (p.copy() + p2.copy()) / 2
                    x = frac * x.copy() + (1 - frac) * x2.copy()

            label_values = np.maximum(label_values, label_values2)

        x = x / x.max()
        assert self.min_clip > 0
        x = np.clip(x, self.min_clip, 1.0)  # todo: check

        if self.scale_before_log > 99:
            x = x / np.max(x) * self.scale_before_log
            x = np.log10(x + 1)
            x = x / np.max(x)
        elif self.log_space:
            x = np.log10(x)

        assert self.norm_to_one
        if self.norm_to_one:
            dbg(x)
            x = norm_min0_max1(x)
            # x = 1.0 + x / abs(np.log10(self.min_clip))
            dbg(x)

        img_tag = self.make_img_tag()
        img_tag = f'{sample_id}_3__getitem__{img_tag}'
        show_sample(cfg=self.cfg, x=x, tag=img_tag)

        x = x.astype(np.float32)
        x_np = x.copy()
        dbg(x)
        if self.output_pytorch_tensor:
            x = x[..., 0:2]  # ignore 3rd
            x = np.moveaxis(x, source=-1, destination=0)
            x = torch.from_numpy(x)
            label_values = torch.from_numpy(label_values)
        res = {
            'item': idx,
            'image': x,
            'sample_id': sample_id,
            'sample_id2': sample_id2,
            'label': label_values,
            'label2': label_values2,
            'derivatized': str(metadata['derivatized'])
        }
        self._sort_save_sample(meta=metadata, x_np=x_np, sample_dict=res)
        return res

    def _sort_save_sample(self, *, meta, x_np, sample_dict):
        if not self.cfg.SORT_SAVE_SAMPLES:
            return

        x = x_np.copy()

        assert self.mix_aug_prob == 0, "Set self.mix_aug_prob=0 if running SORT"
        #         res = {
        #             'item': idx,
        #             'image': x,
        #             'sample_id': sample_id,
        #             'sample_id2': sample_id2,
        #             'label': label_values,
        #             'label2': label_values2,
        #             'derivatized': str(metadata['derivatized'])
        #             # 'instrument_type': metadata['instrument_type']
        #         }
        data_tag = self.make_data_tag()
        if self.normalize_m_separately:
            data_tag = f'{data_tag}_normM{int(self.normalize_m_separately)}'
        if self.normalize_t_separately:
            data_tag = f'{data_tag}_normT{int(self.normalize_t_separately)}'
        sort_save_sample(cfg=self.cfg, meta=meta, x=x, data_tag=data_tag, sample_dict=sample_dict)

        x = x_np.copy()
        dbg(x)
        if len(x.shape) == 3:
            x = np.sum(x, axis=-1, keepdims=True)
            x = np.repeat(x, repeats=3, axis=-1)
        else:
            x = np.repeat(x[:, :, None], repeats=3, axis=-1)

        x = x / x.max()
        x = convert_yaxis_to_ampl(x.copy())
        # idx = (x > 0)
        # x[idx, 0] = 1 - x[idx, 0]
        # x = 1 - x
        dbg(x)
        data_tag = f'{data_tag}_v2'
        sort_save_sample(cfg=self.cfg, meta=meta, x=x, data_tag=data_tag, sample_dict=sample_dict)


# if __name__ == '__main__':
#     set_dbg(True)
#     # check_render_image()
#     # check_dataset()
#     # check_aug()
#     # check_performance()
