import os
import sys
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
import random
from warnings import filterwarnings
filterwarnings("ignore")

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        # print('att1', x.shape) torch.Size([32, 1536, 7])
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        # print('att2', norm_att.shape) torch.Size([32, 152, 7])
        cla = self.nonlinear_transform(self.cla(x))
        # print('att3', cla.shape) torch.Size([32, 152, 7])
        x = torch.sum(norm_att * cla, dim=2)
        # print('att4', x.shape) torch.Size([32, 152])
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'

class NModel(nn.Module):
    def __init__(self,cfg, in_feat):
        super(NModel, self).__init__()
        self.cfg = cfg

        self.spec_augmenter = SpecAugmentation(time_drop_width=2, time_stripes_num=4,
   freq_drop_width=2, freq_stripes_num=4)

        if cfg.rot:
            self.bn0 = nn.BatchNorm2d(cfg.seq_size)
        else:
            self.bn0 = nn.BatchNorm2d(cfg.max_mass)
        
        # self.bn0 = nn.BatchNorm2d(313)

        if self.cfg.stride == 2:
        # if 1:
            self.cbr1 = nn.Conv2d(cfg.in_ch, cfg.in_ch, 3, (2,2))

        # if 'nfnet' in cfg.backbone or 'convnext' in cfg.backbone:
        #     base_model = timm.create_model(
        #         cfg.backbone, pretrained=True, in_chans=cfg.in_ch, 
        #         # act_layer = nn.SiLU
        #         )
        if 'efficient' in cfg.backbone:
            base_model = timm.create_model(
                cfg.backbone, pretrained=True, in_chans=cfg.in_ch, 
                act_layer = nn.SiLU
                )
        else:
            base_model = timm.create_model(
                cfg.backbone, pretrained=True, in_chans=cfg.in_ch, 
                # act_layer = nn.SiLU
                )

        # print(base_model)
        # if 'efficient' in cfg.backbone:
        #     base_model.conv_stem.stride = (cfg.stride, cfg.stride)
        # print('1111===',base_model)

        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        elif 'nfnet' in cfg.backbone:
            in_features = base_model.num_features
        elif 'resne' in cfg.backbone:
            in_features = base_model.fc.in_features
        else:
            # in_features = base_model.classifier.in_features
            in_features = base_model.num_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, len(cfg.target_cols), activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        

    def forward(self, input_data, mask=None):
        x = input_data # (batch_size, 3, time_steps, mel_bins)
        # x = x.transpose(2, 3)
        # print(x.shape)

        if self.cfg.in_ch == 1:
            x = x.unsqueeze(1)

        # print(x.shape)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            if self.cfg.round >6:
                aug_prob = 0.5
            else:
                aug_prob = 0.5

            if random.random() < aug_prob:
                if self.cfg.round >6:
                    spec_augmenter = SpecAugmentation(time_drop_width=random.randint(2,6), time_stripes_num=random.randint(2,6),
       freq_drop_width=random.randint(2,4), freq_stripes_num=random.randint(2,4))
                    x = spec_augmenter(x)
                else:
                    x = self.spec_augmenter(x)

        if self.cfg.stride == 2:
            x = self.cbr1(x)

        frames_num = x.shape[2]

        # x = x.transpose(2, 3)

        x = self.encoder(x)
        # print('1',x.shape)  #torch.Size([4, 2048, 22, 11])
        
        # Aggregate in frequency axis
        # x_m = torch.mean(x, dim=3)
        x,_ = torch.max(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)

        # x1_m = F.max_pool1d(x_m, kernel_size=3, stride=1, padding=1)
        # x2_m = F.avg_pool1d(x_m, kernel_size=3, stride=1, padding=1)

        x = x1 + x2
        # x = x1 + x2 + x1_m + x2_m

        x = F.dropout(x, p=self.cfg.drop_rate, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=self.cfg.drop_rate, training=self.training)

        # print(x.shape) #torch.Size([4, 2048, 20])

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # print('1', segmentwise_logit.shape, segmentwise_output.shape) 
        #1 torch.Size([4, 20, 152]) torch.Size([4, 20, 152])

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                 interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)
        # print(logit.shape, framewise_logit.shape)
        return {
        'out1':logit,
        'out2':framewise_logit
        }
