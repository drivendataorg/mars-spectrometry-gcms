import json
import math
import random
from collections import defaultdict
from pprint import pprint, pformat

import torch, sys
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from timm.models.helpers import resolve_pretrained_cfg
from tqdm import tqdm
from common_utils_v4 import *


class Model_timm(nn.Module):
    def __init__(self):
        super().__init__()
        self.timm_image_size = None
        self.resize_to_timm_input_size = None

    def resizeToTimmSizeIfNeeded(self, x):
        if x.shape[-1] == self.timm_image_size and \
                x.shape[-2] == self.timm_image_size:
            return x
        # assert self.resize_to_timm_input_size, f'timm_image_size={self.timm_image_size}'
        if self.resize_to_timm_input_size:
            out_size = [self.timm_image_size, self.timm_image_size]
            if x.shape[-1] > self.timm_image_size:
                # align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear
                x = F.interpolate(x, size=out_size, mode="area")
            else:
                x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=True)
            dbg(x)
        return x


def run_tta_sizes(*, x, encoder, is_training, size_step, n_steps, h=None, w=None):
    if size_step == 0:
        return encoder(x)
    out = None
    s = size_step
    assert n_steps in [3, 4, 5]
    w_list = [-s, 0, s]
    if n_steps == 2:
        w_list = [0, s]
    if n_steps == 4:
        w_list = [-s, 0, s, 2 * s]
    if n_steps == 5:
        w_list = [-2 * s, -s, 0, s, 2 * s]
    if is_training:  # when train, select only one random resize
        w_list = [random.choice(w_list)]
    if h is None:
        h, w = x.shape[-2:]
    for w1 in w_list:
        new_h = h
        new_w = w + w1
        trans = T.Resize((new_h, new_w))
        new_x = trans(x)
        dbg(new_x)
        new_out = encoder(new_x)
        if out is None:
            out = new_out
        else:
            out = out + new_out
    x = out / (len(w_list))
    dbg(x)
    return x


def run_tta_crops(*, tta_fn, tta_fn_params, x, is_training, crop_step, n_steps, ):
    if crop_step == 0:
        return tta_fn(x)
    out = None
    s = crop_step
    # w_list = []
    assert n_steps in [1, 4, 7]
    w_list = [-s, 0, s]
    if n_steps == 1:
        w_list = [(0, 0)]
    if n_steps == 4:
        w_list = [(0, 0), (s, 0), (0, s), (s, s)]
    if n_steps == 7:
        w_list = [(0, 0), (s, 0), (0, s), (s, s), (2 * s, 0), (0, 2 * s), (2 * s, 2 * s)]
    if is_training:  # when train, select only one random resize
        # w_list = [random.choice(w_list)]
        w_list = [(np.random.randint(2 * s + 1), np.random.randint(2 * s + 1))]
    # else:
    #     w_list = [0]  # DBG
    h, w = x.shape[-2:]
    for w1, w2 in w_list:
        new_x = x[:, :, :, w1: w - w2]  # b, ch, rows, cols
        new_out = tta_fn(x=new_x, **tta_fn_params)  # [-1] for features only
        if out is None:
            out = new_out
        else:
            out = out + new_out
    x = out / (len(w_list))
    dbg(x)
    return x


def make_params(base_model_name, *, in_chans, features_only, pretrained):
    params = dict(model_name=base_model_name, in_chans=in_chans, features_only=features_only,
                  pretrained=pretrained)
    if 'convnext' in base_model_name:
        params['drop_path'] = 0.1
    if 'efficientnet' in base_model_name:
        params['drop_path_rate'] = 0.1
    return params


class Model_1st_SimpleCls3(Model_timm):
    def __init__(self,
                 cfg,
                 base_model_name,
                 resize_to_timm_input_size,
                 embed_rows_cols_2_channels,
                 dropout=0,
                 tta_size_step=0,
                 tta_size_step_num=3,
                 tta_crops_size=0,
                 tta_crops_num=4,
                 image_channels=2,
                 pretrained=True,
                 freeze_base=False,
                 input_to_base_kernel=1,
                 fc_relu=False,
                 ):
        super().__init__()
        n_classes = cfg.n_classes
        self.dropout = dropout
        self.tta_size_step = tta_size_step
        self.tta_size_step_num = tta_size_step_num
        self.tta_crops_size = tta_crops_size
        self.tta_crops_num = tta_crops_num
        self.freeze_base = freeze_base
        self.embed_rows_cols_2_channels = embed_rows_cols_2_channels
        self.image_channels = image_channels
        pretrained_cfg = resolve_pretrained_cfg(base_model_name)
        self.timm_image_size = pretrained_cfg['input_size'][-1]
        self.resize_to_timm_input_size = resize_to_timm_input_size
        if self.embed_rows_cols_2_channels:
            in_chans = image_channels + 2
        else:
            in_chans = image_channels + 1
        base_in_chans = 3

        self.input_to_base = nn.Conv2d(
            in_chans, base_in_chans, kernel_size=input_to_base_kernel,
            padding=(input_to_base_kernel - 1) // 2)
        params = make_params(base_model_name, in_chans=base_in_chans, features_only=False, pretrained=pretrained)
        self.base_model = timm.create_model(**params)
        print(f'{base_model_name}')
        flat_chans = self.base_model.num_classes
        self.drop_before_fc = nn.Dropout(self.dropout)  # note 1d
        self.relu = nn.ReLU()
        self.fc_relu = fc_relu
        self.fc = nn.Linear(flat_chans, n_classes)

    # def forward(self, x):
    #
    #     if self.tta_crops_size > 0:
    #         h, w = x.shape[-2:]
    #         tta_fn_params = dict(encoder=self.forward_one,
    #                              is_training=self.training,
    #                              size_step=self.tta_size_step,
    #                              n_steps=self.tta_size_step_num,
    #                              h=h, w=w,
    #                              )
    #         x = run_tta_crops(tta_fn=run_tta_sizes,
    #                           tta_fn_params=tta_fn_params,
    #                           x=x, is_training=self.training,
    #                           crop_step=self.tta_crops_size,
    #                           n_steps=self.tta_crops_num)
    #     else:
    #         x = run_tta_sizes(encoder=self.forward_one,
    #                           x=x, is_training=self.training,
    #                           size_step=self.tta_size_step,
    #                           n_steps=self.tta_size_step_num)
    #     return x

    def forward(self, x):
        x = run_tta_sizes(encoder=self.forward_one,
                          x=x, is_training=self.training,
                          size_step=self.tta_size_step,
                          n_steps=self.tta_size_step_num)
        return x

    def forward_one(self, x):
        dbg(x)
        # x = self.resizeToTimmSizeIfNeeded(x)

        b, ch, nrows, ncols = x.shape
        xcols = (torch.arange(ncols).float() / ncols).to(x.device)[None, None, None, :]
        xrows = (torch.arange(nrows).float() / nrows).to(x.device)[None, None, :, None]
        xcols = xcols.repeat(b, 1, nrows, 1)
        xrows = xrows.repeat(b, 1, 1, ncols)
        if self.embed_rows_cols_2_channels:
            x = torch.cat([x, xcols, xrows], dim=1)
        else:
            x = torch.cat([x, xcols * xrows], dim=1)

        # assert x.shape[1] == self.image_channels
        if self.freeze_base and self.base_model.training:
            self.base_model.eval()  # todo: freeze on/off
        x = self.input_to_base(x)
        dbg(x)
        x = self.base_model(x)  # note [-1] for features_only=True,
        dbg(x)
        if self.fc_relu:
            x = self.relu(x)
        if self.dropout > 0:
            x = self.drop_before_fc(x)
        x = self.fc(x)
        return x


class Model_cls3flatM(Model_timm):  # , Model_timmLast1k_v2
    def __init__(self,
                 cfg,
                 base_model_name,
                 resize_to_timm_input_size,
                 embed_rows_cols_2_channels,
                 dropout=0,
                 tta_size_step=0,
                 tta_size_step_num=3,
                 m_bins=None,
                 image_channels=2,
                 # reduction_to_flat=8,
                 pretrained=True,
                 freeze_base=False,
                 input_to_base_kernel=1,
                 fc_relu=False,
                 ):
        super().__init__()
        n_classes = cfg.n_classes
        self.dropout = dropout
        self.tta_size_step = tta_size_step
        self.tta_size_step_num = tta_size_step_num
        self.freeze_base = freeze_base
        self.embed_rows_cols_2_channels = embed_rows_cols_2_channels
        self.image_channels = image_channels
        pretrained_cfg = resolve_pretrained_cfg(base_model_name)
        self.timm_image_size = pretrained_cfg['input_size'][-1]
        self.resize_to_timm_input_size = resize_to_timm_input_size
        if self.embed_rows_cols_2_channels:
            in_chans = image_channels + 2
        else:
            in_chans = image_channels + 1
        base_in_chans = 3

        self.input_to_base = nn.Conv2d(
            in_chans, base_in_chans, kernel_size=input_to_base_kernel,
            padding=(input_to_base_kernel - 1) // 2)

        # self.base_model = timm.create_model(
        #     base_model_name, in_chans=base_in_chans, features_only=True, pretrained=pretrained)
        # print(f'{base_model_name}')
        # flat_chans = self.base_model.num_classes
        # self.drop_before_fc = nn.Dropout(self.dropout)  # note 1d
        # self.relu = nn.ReLU()
        # self.fc_relu = fc_relu
        # self.fc = nn.Linear(flat_chans, n_classes)
        self.base_model = timm.create_model(
            base_model_name, in_chans=base_in_chans, features_only=True, pretrained=pretrained)
        self.backbone_depths = list(self.base_model.feature_info.channels())
        print(f'{base_model_name}')
        features_2d_size = m_bins // 32
        feature_chans = self.backbone_depths[-1]
        # reduced_chans = feature_chans // reduction_to_flat
        self.features_to_flat = nn.Sequential(
            # nn.Conv2d(feature_chans, reduced_chans, kernel_size=1),
            nn.Flatten()
        )
        # flat_chans = reduced_chans * features_2d_size
        flat_chans = feature_chans * features_2d_size
        self.drop_before_fc = nn.Dropout(self.dropout)  # note 1d
        self.fc = nn.Linear(flat_chans, n_classes)

    def forward(self, x):
        x = run_tta_sizes(encoder=self.forward_one,
                          x=x, is_training=self.training,
                          size_step=self.tta_size_step,
                          n_steps=self.tta_size_step_num)
        return x

    def forward_one(self, x):
        dbg(x)
        # x = self.resizeToTimmSizeIfNeeded(x)

        b, ch, nrows, ncols = x.shape
        xcols = (torch.arange(ncols).float() / ncols).to(x.device)[None, None, None, :]
        xrows = (torch.arange(nrows).float() / nrows).to(x.device)[None, None, :, None]
        xcols = xcols.repeat(b, 1, nrows, 1)
        xrows = xrows.repeat(b, 1, 1, ncols)
        if self.embed_rows_cols_2_channels:
            x = torch.cat([x, xcols, xrows], dim=1)
        else:
            x = torch.cat([x, xcols * xrows], dim=1)

        # assert x.shape[1] == self.image_channels
        if self.freeze_base and self.base_model.training:
            self.base_model.eval()  # todo: freeze on/off
        x = self.input_to_base(x)
        dbg(x)
        x = self.base_model(x)[-1]  # note [-1] for features_only=True,
        dbg(x)
        x = torch.mean(x, dim=-1)
        dbg(x)
        x = self.features_to_flat(x)
        dbg(x)
        if self.dropout > 0:
            x = self.drop_before_fc(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    main()
