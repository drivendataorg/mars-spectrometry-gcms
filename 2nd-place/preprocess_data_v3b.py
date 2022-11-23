import multiprocessing
from os.path import join, expanduser, basename

import cv2
import pandas as pd
import numpy as np
import os, sys
# import config_v1a as cfg
from config_main_v4 import cfg as cfg
import pybaselines
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import matplotlib.pyplot as plt
from common_utils_v4 import *


def main():
    # set_dbg(True)
    set_dbg(False)
    split_to_folds()


def split_to_folds():
    RANDOM_SEED = 42
    # skf = StratifiedKFold(n_splits=cfg.n_folds, random_state=RANDOM_SEED, shuffle=True)
    skf = MultilabelStratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=RANDOM_SEED)
    # metadata = pd.read_csv("../data/metadata.csv")
    fpath = join(cfg.DATA_DIR, 'metadata.csv')
    dbg('fpath')
    df = pd.read_csv(fpath)
    # sample_id                                     S1583
    # split                                          test
    # derivatized                                     NaN
    # features_path               test_features/S1583.csv
    # features_md5_hash  43ca2f97c4b77d881e228cca0af72887
    # 1584
    dbg(df)
    df.fillna(0, inplace=True)
    df['derivatized'] = df['derivatized'].astype(int)
    df = df[df['split'] == 'train']
    meta = df[['sample_id', 'derivatized']]
    d = df['derivatized'].values
    dbg(d)

    fpath = join(cfg.DATA_DIR, 'train_labels.csv')
    dbg('fpath')
    df = pd.read_csv(fpath)
    # sample_id                      S0808
    # aromatic                           0
    # hydrocarbon                        0
    # carboxylic_acid                    0
    # nitrogen_bearing_compound          0
    # chlorine_bearing_compound          0
    # sulfur_bearing_compound            0
    # alcohol                            0
    # other_oxygen_bearing_compound      0
    # mineral                            0
    # 809
    dbg(df)
    df = pd.merge(df, meta, how='left', on='sample_id')
    dbg(df)
    dbg('list(df.columns)')  # list(df.columns) = ['sample_id', 'aromatic', 'hydrocarbon', 'carboxylic_acid', 'nitrogen_bearing_compound', 'chlorine_bearing_compound', 'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound', 'mineral']
    count_nan = df.isnull().sum().sum()
    assert count_nan == 0, 'FIX nans'

    # df = df[df.split != 'test']
    df['fold'] = -1
    kfold_cols = cfg.cls_labels
    # for fold, (train_index, test_index) in enumerate(skf.split(df.sample_id, df.instrument_type)):
    for fold, (train_index, test_index) in enumerate(skf.split(df, df[kfold_cols])):
        print(fold, test_index)
        df.loc[test_index, 'fold'] = fold

    dbg(df)
    # 0    162
    # 3    162
    # 1    162
    # 4    162
    # 2    161
    print(df['fold'].value_counts())
    # print(df[df.instrument_type == 'sam_testbed']['fold'].value_counts())
    df = df[['sample_id', 'fold']]
    # df[['sample_id', 'fold']].to_csv('../data/folds_v4.csv', index=False)
    # fpath = join(cfg.DATA_DIR, f'folds_n{cfg.n_folds}_seed{RANDOM_SEED}_v1a.csv')
    fpath = join('./', f'folds_n{cfg.n_folds}_seed{RANDOM_SEED}_v1a.csv')
    dbg('fpath')
    if os.path.isfile(fpath):
        print(f'Folds file exists already!!!: {fpath}')
    else:
        df.to_csv(fpath, index=False)


if __name__ == '__main__':
    main()
    # split_to_folds()
    # preprocess_all_features()

