#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import copy
import math
import myswa
import torch
import random
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import log_loss
from shutil import copyfile
from tqdm import tqdm
import numpy as np
import importlib
import argparse
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.append("configs")
sys.path.append("models")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser.add_argument("-M", "--mode", default='train', help="mode type")
parser.add_argument("-A", "--all", default=0, help="use train val")
parser.add_argument("-F", "--nfold", default=5, help="num fold")
parser.add_argument("-S", "--split", default=1, help="split option")
parser_args, _ = parser.parse_known_args(sys.argv)

print("[ √ ] Using config file", parser_args.config)
print("[ √ ] Using mode: ", parser_args.mode)

cfg = importlib.import_module(parser_args.config).cfg
NModel = importlib.import_module(cfg.model).NModel
cfg.mode = parser_args.mode
cfg.all_data = int(parser_args.all)
cfg.num_folds = int(parser_args.nfold)
cfg.split = int(parser_args.split)

cfg.out_dir = f'outputs/{cfg.name}'
cfg.load_weight = cfg.load_weight.replace('/data2/weights/seq_mars/seq', cfg.out_dir.split('/')[0])
# cfg.epochs = 1


if cfg.all_data:
    cfg.use_val = 0

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(cfg.seed)


def logfile(message):
    print(message)
    with open(log_path, 'a+') as logger:
        logger.write(f'{message}\n')

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

import scipy as sp
class NDataset:
    def __init__(self, df,cache_metadata = {}, val=0):
        self.df = df.reset_index(drop=True)
        self.val = val

        self.cache_metadata = cache_metadata

        print(f'==mode val {self.val}', self.df.shape[0])

        
    def __len__(self):
        return (self.df.shape[0])
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        features_path = row['features_path']

        if features_path in self.cache_metadata:
            features = copy.deepcopy(self.cache_metadata[features_path])

            # features1 = cache_metadata1[features_path]

        else:
            features = get_features(row)
        # features = np.random.rand(cfg.seq_size, 32)

        # features = features.T
        # features /= 2000

        # sigma = [3, 3]
        # features = sp.ndimage.filters.gaussian_filter(features, sigma, mode='constant')
        # weights = np.array([[0, 0, 1, 0, 0],
        #             [0, 2, 4, 2, 0],
        #             [1, 4, 8, 4, 1],
        #             [0, 2, 4, 2, 0],
        #             [0, 0, 1, 0, 0]],
        #            dtype=np.float)
        # weights = weights / np.sum(weights[:])
        # features = sp.ndimage.filters.convolve(features, weights, mode='constant')

        # print(features)
        if cfg.is_pretrain:
            features = features*1e16

        if cfg.in_ch == 3:

            if cfg.normalize == 0:
                if features.shape[0] < cfg.seq_size:
                    features = np.concatenate([features, np.zeros((cfg.seq_size-features.shape[0], features.shape[1]))])
                features = features**3

            # if self.val == 0:
            if 0:
                # delta = 
                x1 = 2.5 + 0.01*random.randint(-20,20)
                x2 = 2.0 + 0.01*random.randint(-20,20)
                x3 = 3.0 + 0.01*random.randint(-20,20)

                features1 = features**(1/x1)
                features2 = features**(1/x2)
                features3 = features**(1/x3)
            else:
                if cfg.split == 1:
                    features1 = features**(1/2.5)
                    features2 = features**(1/2)
                    features3 = features**(1/3)
                elif cfg.split == 2:
                    features1 = features**(1/2.2)
                    features2 = features**(1/2.7)
                    features3 = features**(1/3.2)
                else:
                    features1 = features**(1/2.4)
                    features2 = features**(1/2.1)
                    features3 = features**(1/2.8)

            # features2 = np.log2(features)
            # features2 = np.clip(features2, 0,100)

            if self.val == 0:
            # if 0:
                if random.random() < cfg.aug*cfg.aug_prob:
                    # noise = features*0.1*random.random()
                    noise = random.randint(-100,100)
                    features1 = features1 + noise

                    noise = random.randint(-100,100)
                    features2 = features2 + noise

                    noise = random.randint(-100,100)
                    features3 = features3 + noise
                elif cfg.aug == 2:
                    noise = random.randint(-100,100)
                    features1 = features1 + noise
                    if random.random()>0.5: #random drop out
                        # features1 *= 0.01*random.randint(80,120)
                        mask = 1*(np.random.rand(features1.shape[0],features1.shape[1])>0.2)
                        features1= features1*mask


                    noise = random.randint(-100,100)
                    features2 = features2 + noise
                    if random.random()>0.5: #random drop out
                        # features2 *= 0.01*random.randint(80,120)
                        mask = 1*(np.random.rand(features2.shape[0],features2.shape[1])>0.2)
                        features2= features2*mask

                    noise = random.randint(-100,100)
                    features3 = features3 + noise
                    if random.random()>0.5: #random drop out
                        # features3 *= 0.01*random.randint(80,120)
                        mask = 1*(np.random.rand(features3.shape[0],features3.shape[1])>0.2)
                        features3= features3*mask

            # features = np.concatenate([features2, features3], -1)
            features = np.stack([features1[:cfg.seq_size,:cfg.max_mass], features2[:cfg.seq_size,:cfg.max_mass], features3[:cfg.seq_size,:cfg.max_mass]])
            # print(features.shape, features1.shape)
            mask = np.ones(features.shape[0])

            # features /= 2000

        elif cfg.in_ch == 4:

            if cfg.normalize == 0:
                if features.shape[0] < cfg.seq_size:
                    features = np.concatenate([features, np.zeros((cfg.seq_size-features.shape[0], features.shape[1]))])
                features = features**3

            features1 = features**(1/2.5)
            features2 = features**(1/2)
            features3 = features**(1/3)


            if self.val == 0:
            # if 0:
                if random.random() < cfg.aug*cfg.aug_prob:
                    # noise = features*0.1*random.random()
                    noise = random.randint(-100,100)
                    features1 = features1 + noise

                    noise = random.randint(-100,100)
                    features2 = features2 + noise

                    noise = random.randint(-100,100)
                    features3 = features3 + noise

            features4 = copy.deepcopy(cache_metadata_mean[features_path])
            # features = np.concatenate([features2, features3], -1)
            features = np.stack([features1[:cfg.seq_size,:cfg.max_mass], features2[:cfg.seq_size,:cfg.max_mass], features3[:cfg.seq_size,:cfg.max_mass], features4[:cfg.seq_size,:cfg.max_mass]])
            # print(features.shape, features1.shape)
            mask = np.ones(features.shape[0])

        elif cfg.in_ch == 5:

            if cfg.normalize == 0:
                if features.shape[0] < cfg.seq_size:
                    features = np.concatenate([features, np.zeros((cfg.seq_size-features.shape[0], features.shape[1]))])
                features = features**3

            features1 = features**(1/1.8)
            features2 = features**(1/2.2)
            features3 = features**(1/2.6)
            features4 = features**(1/3.0)
            features5 = features**(1/3.4)

            # features2 = np.log2(features)
            # features2 = np.clip(features2, 0,100)

            if self.val == 0:
            # if 0:
                if random.random() < cfg.aug*cfg.aug_prob:
                    # noise = features*0.1*random.random()
                    noise = random.randint(-100,100)
                    features1 = features1 + noise

                    noise = random.randint(-100,100)
                    features2 = features2 + noise

                    noise = random.randint(-100,100)
                    features3 = features3 + noise

                    noise = random.randint(-100,100)
                    features4 = features4 + noise

                    noise = random.randint(-100,100)
                    features5 = features5 + noise

            # features = np.concatenate([features2, features3], -1)
            features = np.stack([features1[:cfg.seq_size,:cfg.max_mass], features2[:cfg.seq_size,:cfg.max_mass], features3[:cfg.seq_size,:cfg.max_mass], features4[:cfg.seq_size,:cfg.max_mass], features5[:cfg.seq_size,:cfg.max_mass]])
            # print(features.shape, features1.shape)
            mask = np.ones(features.shape[0])
        else:

            # features1 = np.zeros((features.shape[0], 200))
            # for i in range(200):
            #     features1[:,i] = features[:,2*i] + features[:,2*i+1]
            # features = features1

            # if self.val == 0:
            # if 0:
            #     pow_r = 0.1*random.randint(23,27)
            # else:
            #     pow_r = 3.5

            if cfg.normalize == 1:
                power_r = cfg.power_r
                if self.val == 0 and random.random() < cfg.power_prob:
                    power_r += 0.01*random.randint(-20,20)

                if cfg.use_meta:
                    features[:,:-2] = features[:,:-2]**(1/power_r)
                else:
                    features = features**(1/power_r)

            # features = np.log2(features)
            # features = np.clip(features, 0,100)
            features = features[:cfg.seq_size,:cfg.max_mass]

            # for i in range(features.shape[-1]):
            #     features[:,i] = features[:,i]/np.mean(features[:,i])

            # features = np.clip(features, 0,100)

            # features = minmax_scale(features.reshape(-1)).reshape(cfg.seq_size,-1)


            # features = features / np.max(features)

            features = features[:cfg.seq_size, :]

            # features1 = features1[:cfg.seq_size, :]
            # features = np.concatenate([features, features1], -1)

            # features1 = copy.deepcopy(features)
            # for i in range(features1.shape[-1]-2):
            #     if i == 0:
            #         features1[:,i] = features[:,i] + features[:,i+1]
            #     elif i == features1.shape[-1] - 3:
            #         features1[:,i] = features[:,i] + features[:,i-1]
            #     else:
            #         features1[:,i] = features[:,i] + features[:,i+1] + features[:,i-1]
            # features = features1

            if self.val == 0 and random.random() < cfg.aug*cfg.aug_prob:
            # if 0:
                # noise = features*0.1*random.random()
                noise = random.randint(-100,100)
                if cfg.use_meta:
                    features[:,:-2] = features[:,:-2] + noise
                else:
                    features = features + noise

                # if random.random()>0.5: #random drop out
                #     mask = 1*(np.random.rand(features.shape[0],features.shape[1])>0.1)
                #     features= features*mask

            # if cfg.normalize:
            #     features[:,:-2] = features[:,:-2]/500

            # features[:,:-2] /= np.max(features[:,:-2])
            # features[:,:-2] = np.clip(features[:,:-2], 1e-5, 1.0)
            # features[:,:-2] = np.log10(features[:,:-2])

            # features /= 300

            # features[:,:-2] = features[:,:-2]**2

            # features = features[:,:200]

            # print(features[:,-2])

            mask = np.ones(features.shape[0])
            if features.shape[0] < cfg.seq_size:
                # features = np.concatenate([features, np.zeros_like(features[-(cfg.seq_size-features.shape[0]):])])
                features = np.concatenate([features, np.zeros((cfg.seq_size-features.shape[0], features.shape[1]))])
                mask = np.concatenate([mask, np.zeros((cfg.seq_size-mask.shape[0]))])

            # features = (features.T/np.max(features,1)).T
            if cfg.model == 'model_sed':
                pass
            else:
                features = features.transpose(1,0)

        # print(features.shape)
        if cfg.rot:
            features = features.transpose(0,2,1)
        # print(features.shape)

        if cfg.scale_aug and self.val == 0:
            ratio = 0.01*random.randint(70,120)
            features*=ratio

        targets = np.array(row[cfg.target_cols], dtype=float)
        # print(targets)

        # if self.val == 0:
        #     targets = np.clip(targets, 0.0025, 0.9975)

        dct = {
            'x' : torch.tensor(features, dtype=torch.float),
            'y' : torch.tensor(targets, dtype=torch.float),
            'm' : torch.tensor(mask, dtype=torch.float),
        }
        return dct

def val_func(model, valid_loader, get_emb = False):
    model.eval()
    val_losses = []
    pred = []
    y_trues = []
    embeddings = []
    for data in tqdm(valid_loader):
        x,y = data['x'].to(device),data['y'].to(device)
        mask = data['m'].to(device)
        with torch.no_grad():
            pred_out = model(x, mask)

        outputs = pred_out['out1']#.squeeze(-1)
        y_trues.append(y.detach().cpu().numpy())
        if cfg.loss_type == 'bce':
            out = outputs.sigmoid().detach().cpu().numpy()
            pred.append(out)
            loss = loss_bce_fn(outputs, y)
        elif cfg.loss_type == 'dual':
            out = outputs.sigmoid().detach().cpu().numpy()
            out1 = pred_out['out2'].sigmoid().detach().cpu().numpy()
            out = 0.5*out + 0.5*out1
            pred.append(out)
            loss = loss_bce_fn(outputs, y)

            loss1 = loss_bce_fn(pred_out['out2'], y)
            loss = 0.5*loss +0.5*loss1
        elif cfg.loss_type == 'focal':
            loss = loss_bce_fn(pred_out, y)
            out = outputs.sigmoid().detach().cpu().numpy()

            if len(en_suf)>1:
                # print('aaa=====')
                output_with_max, _ = pred_out['out2'].max(dim=1)
                output_with_max = output_with_max.sigmoid().detach().cpu().numpy()
                out = 0.5*out + 0.5*output_with_max

            pred.append(out)

        val_losses.append(loss.item())

        if get_emb:
            embeddings.append(pred_out['emb'].detach().cpu().numpy())

    pred = np.concatenate(pred).astype(np.float64)
    y_trues = np.concatenate(y_trues).astype(np.float64)

    # pred = np.round(pred,6)
    # print(np.sum(1*(np.isnan(pred.any()))))
    # print(np.sum(1*(np.isfinite(pred.any()))))

    # a, b = np.where(np.isfinite(pred))
    # print(a.shape, b.shape)
    # print(pred[a[0],b[0]])
    # print(pred[:10])

    if get_emb:
        embeddings = np.concatenate(embeddings)

    print(pred.shape, y_trues.shape)
    # y_trues = y_trues.reshape(-1, window_size)
    # pred = pred.reshape(-1, window_size)
    # print(pred.shape, y_trues.shape)

    # print(y_trues[:10], pred[:10])
    if cfg.mode == 'test':
        acc, auc, metric = 0,0,0
    else:
        acc = accuracy_score(y_trues, pred > 0.5)
        auc = roc_auc_score(y_trues, pred)

        metric = []
        for i in range(len(cfg.target_cols)):
            metric.append(log_loss(y_trues[:,i], pred[:,i]))
        print(metric)
        metric = np.mean(metric)
    
    return acc, auc, np.mean(val_losses), metric, pred, y_trues, embeddings

import cv2
def vil_func(model, valid_loader, get_emb = False):
    colors = [(255,160,122), (250,128,114), (220,20,60), (139,0,0), (255,215,0), (255,140,0), (255,255,224), (189,183,107), (124,252,0), (50,205,50), (34,139,34),
            (0,100,0), (173,255,47), (0,250,154), (0,255,255), (0,139,139), (65,105,225), (0,0,255), (106,90,205), (218,112,214), (255,0,255), (75,0,130)]
    model.eval()
    val_losses = []
    pred = []
    y_trues = []
    embeddings = []
    for b_idx,data in enumerate(tqdm(valid_loader)):
        x,y = data['x'].to(device),data['y'].to(device)
        mask = data['m'].to(device)
        with torch.no_grad():
            pred_out = model(x, mask)

        clip_outputs = pred_out['out1'].sigmoid().detach().cpu().numpy()
        frame_outputs = pred_out['out2'].sigmoid().detach().cpu().numpy()

        for i in range(clip_outputs.shape[0]):
            index = b_idx*cfg.train_bs + i
            row = val_df.loc[index]
            root = row['features_path'].split('.')[0].split('/')[-1]
            clip_output = clip_outputs[i]
            frame_output = frame_outputs[i].T
            # print(frame_output.shape, clip_output.shape)

            indexs = list(np.where(clip_output>0.2)[0])
            graph = 50*np.ones((200, frame_output.shape[1], 3))
            cc = 1
            for idx in indexs:
                arr = 1-frame_output[idx]
                # print(len(arr))
                for ii in range(len(arr)):
                    this_v = int(arr[ii]*199)
                    if ii>0 and ii<len(arr)-1:
                        prev_v = int(arr[ii-1]*199)
                        y1 = min(this_v, prev_v)
                        y2 = max(this_v, prev_v)
                        graph[y1:y2,ii] = colors[idx%len(colors)]

                    graph[this_v,(ii):((ii+1))] = colors[idx%len(colors)]

                cv2.putText(graph, f"---{idx}", (10, 20*cc), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[idx%len(colors)],1,cv2.LINE_AA)
                cc+=1

            cv2.putText(graph, f"{row[cfg.target_cols].values}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1,cv2.LINE_AA)
            cv2.imwrite(f'draw/{root}.jpg', graph)

def get_scheduler(optimizer, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup,
            num_training_steps=total_steps,
        )

    return scheduler

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets, masks=None):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1. - probas)**self.gamma * bce_loss + (1. - targets) * probas**self.gamma * bce_loss
        # print(loss.shape)
        # if cfg.use_masks:
        if 0:
            loss = loss*masks
            loss = loss.sum()/(masks.sum())
        else:
            loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target, masks=None):
        input_ = input["out1"]
        target = target.float()

        framewise_output = input["out2"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target, masks)
        aux_loss = self.focal(clipwise_output_with_max, target, masks)

        # if cfg.round == 8:
            # print('a')
            # return self.weights[0] * loss
        return self.weights[0] * loss + self.weights[1] * aux_loss

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

if __name__ == '__main__':
    csv_suf = ''
    if cfg.split == 2:
        csv_suf = '_s2'
        print('=== using seed 2!!!!')
        cfg.cache_path = cfg.cache_path[:-4] + '_s.npy'
        # cfg.power_r = 2.8
        print(cfg.cache_path)
    elif cfg.split == 3:
        csv_suf = '_m'
        print('=== using seed 2!!!!')
        cfg.cache_path = cfg.cache_path[:-4] + '_mean.npy'
        # cfg.power_r = 2.8
        print(cfg.cache_path)

    # en_suf = '_e'
    en_suf = ''

    if cfg.is_pretrain:
        df = pd.read_csv('../data/supplemental_metadata.csv')
        df = df.head(1494)
        df = df.fillna(0)
    else:
        if cfg.num_folds == 10:
            df = pd.read_csv(f'../data/interim/train_10folds{csv_suf}.csv')
        else:
            df = pd.read_csv(f'../data/interim/train_folds{csv_suf}.csv')
        df = df.fillna(0)

    os.makedirs(cfg.out_dir, exist_ok=True)

    if cfg.mode == 'train':
        copyfile(os.path.basename(__file__), os.path.join(cfg.out_dir, os.path.basename(__file__)))
        copyfile(f'configs/{parser_args.config}.py', os.path.join(cfg.out_dir, f'{parser_args.config}.py'))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if os.path.isfile(cfg.cache_path):
        cache_metadata = np.load(cfg.cache_path, allow_pickle=True).item()
    else:
        # cache data for faster training
        features_list = Parallel(n_jobs=16, backend="threading")(delayed(get_features)(row) for i, row in tqdm(df.iterrows()))
        cache_metadata = {path: feat for path, feat in zip(df.features_path.values, features_list)}
        np.save(cfg.cache_path, cache_metadata)

    if cfg.mode == 'test':
        val_cache_metadata = np.load(cfg.cache_path.replace('train_', 'val_'), allow_pickle=True).item()
        test_cache_metadata =  np.load(cfg.cache_path.replace('train_', 'test_'), allow_pickle=True).item()
        # test_cache_metadata = val_cache_metadata

        df = pd.read_csv('../data/raw/metadata.csv')
        sub_df = pd.read_csv('../data/raw/submission_format.csv')
        test_df = df[df.split != 'train']
        # test_df = df[df.split == 'val']
        print(test_df.shape, sub_df.shape)
        for k,val in val_cache_metadata.items():
            test_cache_metadata[k] = val

        for col in cfg.target_cols:
            test_df[col] = random.randint(0,1)

        test_dataset = NDataset(test_df, test_cache_metadata, val=1)

        test_loader = DataLoader(test_dataset,
                                  batch_size=cfg.train_bs,
                                  shuffle=False,
                                  num_workers=0, 
                                  pin_memory=True, 
                                  drop_last=False,
                                  )

    if cfg.use_val or cfg.all_data:
        val_cache_metadata = np.load(cfg.cache_path.replace('train_', 'val_'), allow_pickle=True).item()
        for k,val in val_cache_metadata.items():
            cache_metadata[k] = val

    num_epochs = cfg.epochs
    train_folds = cfg.folds
    if cfg.num_folds == 10:
        train_folds = [0,1,2,3,4,5,6,7,8,9,10]
    elif cfg.all_data:
        train_folds =  [0,1,2,3,4,5]

    cfg.use_val = 0
    # if 'b7_bin005_sed_3ch' in parser_args.config:
    #     train_folds =  [1]

    oof_pred = []
    oof_true = []
    print(train_folds)
    for cc, fold in enumerate(train_folds):
        # model = NModel(cfg, cfg.max_mass//cfg.bin_size + 2)
        model = NModel(cfg, cfg.feat_size)

        model.to(device)

        log_path = f'{cfg.out_dir}/log_f{fold}.txt'

        if cfg.mode != 'test':
            if cfg.is_pretrain:
                # train_df = df.head(647)
                # val_df = df.tail(162)
                train_df = df.head(1200)
                val_df = df.tail(294)
            else:
                if fold == cfg.num_folds:
                    train_df = df
                    val_df = df[df.fold==0]
                else:
                    train_df = df[df.fold!=fold]
                    val_df = df[df.fold==fold]

            if cfg.use_val:
                val_df1 = pd.read_csv(f'../data/interim/val_folds{csv_suf}.csv')
                val_df1 = val_df1.fillna(0)
                # train_df = pd.concat([train_df, val_df1.head(112)])
                # val_df = val_df1.tail(200)
                val_df = val_df1

            if cfg.all_data:
                if cfg.num_folds == 10:
                    df1 = pd.read_csv(f'../data/interim/val_10folds{csv_suf}.csv')
                else:
                    df1 = pd.read_csv(f'../data/interim/val_folds{csv_suf}.csv')
                df1 = df1.fillna(0)
                if fold == cfg.num_folds:
                    train_df1 = df1
                    val_df1 = df1[df1.fold==0]
                else:
                    train_df1 = df1[df1.fold!=fold]
                    val_df1 = df1[df1.fold==fold]
                train_df = pd.concat([train_df, train_df1])
                val_df = pd.concat([val_df, val_df1]).reset_index(drop=True)

            train_df = train_df.reset_index(drop=True)
            train_dataset = NDataset(train_df, cache_metadata)

            # if cfg.mode == 'train':
            #     val_df = val_df.head(50)

            valid_dataset = NDataset(val_df, cache_metadata, val=1)

            train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.train_bs,
                                  shuffle=True,
                                  # sampler=SimpleBalanceClassSampler(train_df),
                                  num_workers=0, 
                                  pin_memory=True, 
                                  drop_last=True,
                                  )
            valid_loader = DataLoader(valid_dataset,
                                      batch_size=cfg.train_bs,
                                      shuffle=False,
                                      num_workers=0, 
                                      pin_memory=True, 
                                      drop_last=False,
                                      )

        if cfg.is_sed:
            loss_bce_fn = BCEFocal2WayLoss()
        else:
            loss_bce_fn = nn.BCEWithLogitsLoss()
            # loss_bce_fn = BCEFocalLoss()

        use_ckpt = 'last'
        # if fold == 3 and parser_args.config == 'b7_bin005_sed_3ch_r7':
        #     use_ckpt = 'best'

        # if '_10f' in parser_args.config:
        #     use_ckpt = 'best'


        if cfg.mode == 'train':
            logfile(f'===== training model {fold+1}/{len(train_folds)} ======')
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=1e-6, betas=(0.9, 0.99))

            if cfg.use_swa:
                optimizer = myswa.SWA(
                    optimizer,
                    swa_start=100,
                    swa_freq=30,
                    swa_lr=None)

            scheduler = get_scheduler(optimizer, num_epochs*len(train_loader))

            if len(cfg.load_weight) > 10:
                if '.pth' in cfg.load_weight:
                    load_weight = cfg.load_weight
                else:
                    if cfg.all_data:
                        load_weight = f'{cfg.load_weight}/last_f{fold}_a{csv_suf}.pth'
                    else:
                        load_weight = f'{cfg.load_weight}/last_f{fold}{csv_suf}.pth'

                logfile(f'load pretrained weight {load_weight}!!!')
                # model.load_state_dict(torch.load(cfg.load_weight))
                state_dict = torch.load(load_weight, map_location=device)  # load checkpoint
                state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
                model.load_state_dict(state_dict, strict=False)  # load
                logfile('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), load_weight))  # report
                del state_dict
                gc.collect()

            not_improve_epochs = 0
            best_score = 0
            best_loss = 10
            best_metric = 10
            global_step = 0
            best_pred = 0
            best_true = 0
            for epoch in range(num_epochs):
                model.train()
                train_losses = []
                tk0 = tqdm(train_loader, total=len(train_loader))
                for bi, data in enumerate(tk0):
                    optimizer.zero_grad()
                    x,y = data['x'].to(device),data['y'].to(device)
                    mask = data['m'].to(device)

                    #mixup
                    if random.random() < cfg.mixup_prob:
                        indices = torch.randperm(x.size(0))
                        shuffled_data = x[indices]
                        shuffled_targets = y[indices]

                        # if random.random()<0.5:
                        if 1:
                            x = 0.5*x + 0.5*shuffled_data
                        else:
                            x = torch.max(x,shuffled_data)

                        y = torch.max(y, shuffled_targets)
                    #~mixup


                    pred_out = model(x, mask)
                    outputs = pred_out['out1']#.squeeze(-1)
                    if cfg.loss_type == 'bce':
                        loss = loss_bce_fn(outputs, y)
                    elif cfg.loss_type == 'dual':
                        loss = loss_bce_fn(outputs, y)
                        loss1 = loss_bce_fn(pred_out['out2'], y)
                        loss = 0.5*loss +0.5*loss1
                    elif cfg.loss_type == 'focal':
                        loss = loss_bce_fn(pred_out, y)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    train_losses.append(loss.item())

                    global_step +=1
                    if cfg.use_swa and global_step%100 == 0 and global_step>100:
                        optimizer.update_swa()

                    tk0.set_description(f'[TRAIN] Fold {fold+1}/{len(train_folds)} Epoch {epoch+1}/{num_epochs}  loss: {np.mean(train_losses):.5f}, LR {scheduler.get_lr()[0]:.6f}')


                logfile(f'[TRAIN] EPOCH {epoch+1} TRAIN LOSS {np.mean(train_losses)}')
                # eval
                if cfg.use_swa:
                    optimizer.swap_swa_sgd()
                acc, auc, val_loss, metric, y_pred, y_true,_ = val_func(model, valid_loader)
                if cfg.use_swa:
                    optimizer.swap_swa_sgd()
                if metric<=best_metric:
                    logfile(f'best log loss improve from {best_metric} to {metric} !!!!!!')
                    best_metric = metric
                    best_pred = y_pred
                    best_true = y_true
                    # if cfg.all_data:
                    #     torch.save(model.state_dict(), f'{cfg.out_dir}/best_f{fold}_a{csv_suf}.pth')
                    # else:
                    #     torch.save(model.state_dict(), f'{cfg.out_dir}/best_f{fold}{csv_suf}.pth')

                logfile(f'VAL EP {epoch+1} acc {acc:.4f} auc {auc:.4f} loss {val_loss:.4f} log loss {metric:.4f}')

            if cfg.all_data:
                torch.save(model.state_dict(), f'{cfg.out_dir}/last_f{fold}_a{csv_suf}.pth')
            else:
                torch.save(model.state_dict(), f'{cfg.out_dir}/last_f{fold}{csv_suf}.pth')

            for i in range(9):
                val_df[f'pred{i}'] = best_pred[:,i]
            oof_pred.append(val_df)

            del model, scheduler, optimizer
            gc.collect()
        elif cfg.mode == 'val':
            if cfg.all_data:
                model_path = f'{cfg.out_dir}/{use_ckpt}_f{fold}_a{csv_suf}.pth'
            else:
                model_path = f'{cfg.out_dir}/{use_ckpt}_f{fold}{csv_suf}.pth'

            print(model_path)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            acc, auc, val_loss, metric, y_pred, y_true, embeddings = val_func(model, valid_loader, 0)

            for i in range(9):
                val_df[f'pred{i}'] = y_pred[:,i]

            oof_pred.append(val_df)

            print(f'VAL fold {fold} acc {acc:.4f} auc {auc:.4f} loss {val_loss:.4f} log loss {metric:.4f}')
            del model 
            gc.collect()
            if fold == cfg.num_folds - 1:
                break
        elif cfg.mode == 'test':
            print(f'===== predict fold {fold}')
            if cfg.all_data:
                model_path = f'{cfg.out_dir}/{use_ckpt}_f{fold}_a{csv_suf}.pth'
            else:
                model_path = f'{cfg.out_dir}/{use_ckpt}_f{fold}{csv_suf}.pth'

            print(model_path)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            _, _, _, _, y_pred, _, _ = val_func(model, test_loader, 0)
            if cc == 0:
                test_pred = y_pred
            else:
                test_pred += y_pred

            del model 
            gc.collect()
        elif cfg.mode == 'vil':
            if cfg.all_data:
                model_path = f'{cfg.out_dir}/{use_ckpt}_f{fold}_a{csv_suf}.pth'
            else:
                model_path = f'{cfg.out_dir}/{use_ckpt}_f{fold}{csv_suf}.pth'

            print(model_path)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            vil_func(model, valid_loader, 0)
            del model 
            gc.collect()

    if cfg.mode == 'test':
        test_pred = test_pred/len(train_folds)
        for i, col in enumerate(cfg.target_cols):
            test_df[col] = test_pred[:,i]
        out_cols = list(sub_df.columns)
        test_df = test_df[out_cols]
        sub_df = sub_df[['sample_id']].merge(test_df, on = ['sample_id'])
        print(sub_df.shape)
        print(sub_df.head())
        if cfg.all_data:
            sub_df.to_csv(f'{cfg.out_dir}/test_a{csv_suf}{en_suf}.csv', index=False)
        else:
            sub_df.to_csv(f'{cfg.out_dir}/test{csv_suf}{en_suf}.csv', index=False)
    else:
        oof_pred = pd.concat(oof_pred)

        print(oof_pred.shape)

        pred_cols = []
        for i in range(9):
            pred_cols.append(f'pred{i}')

        y_pred = oof_pred[pred_cols].values.astype(np.float64) 
        y_true = oof_pred[cfg.target_cols].values.astype(np.float64) 

        acc = accuracy_score(y_true, y_pred > 0.5)
        auc = roc_auc_score(y_true, y_pred)

        metric = []
        for i in range(len(cfg.target_cols)):
            metric.append(log_loss(y_true[:,i], y_pred[:,i]))
        print(metric)
        metric = np.mean(metric)

        logfile(f'OOF acc {acc:.4f} auc {auc:.4f} log loss {metric:.4f}!!!')
        if cfg.all_data:
            oof_pred.to_csv(f'{cfg.out_dir}/oof_a{csv_suf}{en_suf}.csv', index=False)
        else:
            oof_pred.to_csv(f'{cfg.out_dir}/oof{csv_suf}{en_suf}.csv', index=False)
