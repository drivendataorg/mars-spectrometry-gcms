import numpy as np 
from sklearn.metrics import log_loss
import pandas as pd


paths = [

    f'outputs/b5_bin005_sed_3ch_r6/', #0.1520
    'outputs/b5_bin005_sed_3ch_r7', #0.1558

    'outputs/b6_bin005_sed_3ch_r6', #0.1456
    'outputs/b6_bin005_sed_3ch_r7', #0.1474

    'outputs/b7_bin005_sed_3ch_r6', #0.1505
    'outputs/b7_bin005_sed_3ch_r7', #0.1476
    
    f'outputs/m13_bin005_r9/', #0.1435
    f'outputs/m13_bin005_r10/', #0.1444

    f'outputs/m13_bin006_r6', #0.1444
    f'outputs/m13_bin006_r7', #0.1463
    f'outputs/m13_bin006_r8', #0.1465

    f'outputs/m10_bin005_v1_r5/', #0.1517
    f'outputs/m10_bin005_v1_r6/', #0.1513
    f'outputs/m10_bin005_v1_r7/', #0.1509

    f'outputs/b5_bin005_sed_3ch_v1_r6/',
    f'outputs/b5_bin005_sed_3ch_v1_r7/',

    f'outputs/b5_bin005_sed_3ch_v1_r5///',
    f'outputs/b5_bin005_sed_3ch_v1_r6///',
    f'outputs/b5_bin005_sed_3ch_v1_r7///',

    'outputs/m13_bin006_r7///', #0.1458
    f'outputs/m13_bin006_r8///', #0.1446

    'outputs/m13_1_bin006_r6/',
    f'outputs/m13_1_bin006_r7/',
    f'outputs/m13_1_bin006_r8/',

]

target_cols = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']

print(len(paths))

suf = '_a'
# suf = ''

df = pd.read_csv('../data/interim/train_folds.csv')
if suf == '_a':
    df1 = pd.read_csv(f'../data/interim/val_folds.csv')
    df = pd.concat([df, df1]).reset_index(drop=True)
    print(df.shape)

pred_cols = []
for i in range(9):
    pred_cols.append(f'pred{i}')
for i, path in enumerate(paths):
    csv_suf = ''
    if '///' in path:
        csv_suf = '_s2'
    if i == 0:
        pred = pd.read_csv(f'{path}/oof{suf}{csv_suf}.csv')
        pred = df[['sample_id']].merge(pred, on=['sample_id'])
        # pred = pred.head(312)
        # print(pred.shape)
    else:
        pred1 = pd.read_csv(f'{path}/oof{suf}{csv_suf}.csv')
        # pred1 = pred1.head(1121)
        pred1 = df[['sample_id']].merge(pred1, on=['sample_id'])
        pred[pred_cols] += pred1[pred_cols]
        # print(pred1.shape)

pred[pred_cols] /= len(paths)
gt = df[target_cols].values#.astype(np.float64)
pred = pred[pred_cols].values
print(gt.shape, pred.shape)
metric = []
for i in range(9):
    metric.append(log_loss(gt[:,i], pred[:,i]))
metric = np.mean(metric)
print(metric)


# modified from https://github.com/drivendataorg/mars-spectrometry/blob/main/2nd%20Place/TrainEnsemble.py

all_preds = {}
for i, path in enumerate(paths):
    csv_suf = ''
    if '///' in path:
        csv_suf = '_s2'
    # pred = np.load(f'{path}/pred.npy')
    pred = pd.read_csv(f'{path}/oof{suf}{csv_suf}.csv')
    pred = df[['sample_id']].merge(pred, on=['sample_id'])
    # print(i, pred.shape)
    all_preds[f'm{i}'] = pred[pred_cols].values 

all_pred_dfs = {}
for target in range(9):
    all_pred_dfs[target] = pd.DataFrame({k: all_preds[k][:,target] 
                                for k in sorted(all_preds.keys())})

y = pd.DataFrame(gt, index=range(gt.shape[0]),columns=range(9))


#TEST

t_cols = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']
target_cols = {i:x for i,x in enumerate(t_cols)} 
all_preds = {}
for i, path in enumerate(paths):
    csv_suf = ''
    if '///' in path:
        csv_suf = '_s2'
    pred = pd.read_csv(f'{path}/test{suf}{csv_suf}.csv')
    pred = pred[t_cols]
    # print(i, pred.shape)
    if suf == '':
        pred = pred.head(312)
    all_preds[f'm{i}'] = pred.values#[:,1:] 

all_test_dfs = {}
for target in range(9):
    all_test_dfs[target] = pd.DataFrame({k: all_preds[k][:,target] 
                                for k in sorted(all_preds.keys())})
#~TEST

def score(wts, x, y, reg = 1, l1_ratio = 0):
    # wsum = wts.sum()
    wts = (wts / max(wts.sum() ** 0.5, 1.0) )#.astype(np.float32)
    blend = ( x * wts[None, :]).sum(axis = 1)#.astype(np.float32)
    # print(wts)
    return ( 
        log_loss(y, logit(blend))
            + reg *( (wts ** 2).sum() + l1_ratio * np.abs(wts).sum()) )


def optimize(x, y, reg = 1, l1_ratio = 0, tol = 1e-4 ):
    wts = scipy.optimize.minimize(
    score, np.ones(x.shape[1]) / x.shape[1],#len(x.columns), 
        tol = tol,
    args=(x, y, reg, l1_ratio), 
    bounds=[(0, 1) for i in range(x.shape[1])],#len(x.columns))],
    ).x
    return wts / max(wts.sum() ** 0.5, 1.0)


import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import random, datetime, time
import scipy
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
random.seed(datetime.datetime.now().microsecond)

class MultiLabelStratifiedKFold:
    def __init__(self, n_splits=5, random_state=None, full = False):
        self.n_splits = n_splits
        self.random_state = random_state
        self.full = full
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
        
    def split(self, X, y=None, groups=None, verbose = 0):
        n_cols = y.shape[1]
        if self.full:
            y_full = pd.DataFrame(0, y.index, 
                                  np.arange(0, y.shape[1] * (y.shape[1] +1  ) //2), dtype =np.float32)
            y_full.iloc[:, :y.shape[1]] = y.copy()
            ctr = y.shape[1]
            for i in range(y.shape[1]):
                for j in range(y.shape[1]):
                    if i >= j: continue;
                    y_full.iloc[:, ctr] = y.iloc[:, i] * y.iloc[:, j]; ctr +=1
            y = y_full
        freq = y.mean(axis = 0 ).values
        fold_ids = [[] for i in range(self.n_splits)]
        folds = [ np.zeros(y.shape) for i in range(self.n_splits)]
        lwt =  (len(y) / 5) ** -0.5 / ( 1 + y.sum(axis = 1).mean())
        y = y.sample(frac = 1, random_state = self.random_state).astype(np.int32)
        # print(y)
        def safeLog(x, eps = 1e-2): return np.log(np.array(x) * (1-  eps) + eps)
        
        for sample_id, row in zip(y.index, y.values):
        # for sample_id, row in y.sample(frac = 1, random_state = self.random_state).iterrows():
            for idx in range(len(folds)):
                folds[idx][len(fold_ids[idx])] = row#.values
            cts = [np.sum(f[:len(fold_ids[idx])], axis = 0) for idx, f in enumerate(folds)]
            pre_means = [cts[idx] / len(fold_ids[idx]) 
                                if len(fold_ids[idx]) > 0 else np.zeros(y.shape[1]) 
                         for idx, f in enumerate(folds)]
            means = [(cts[idx] + row ) / (len(fold_ids[idx]) +1)
                                   for idx, f in enumerate(folds)]
            pre_scores = ( ( safeLog(pre_means) - safeLog(freq) )** 2).sum(axis = 1) 
            post_scores = ( ( safeLog(means) - safeLog(freq)) ** 2).sum(axis = 1)
            # print(pre_scores)
            # print(post_scores)
            psd =  pre_scores.std() 
            delta_score = post_scores - pre_scores + ( [ psd
                                  * lwt *
                                  ( len(fold_ids[idx]) 
                                       + f[:len(fold_ids[idx]), :n_cols].sum() )
                                       for idx, f in enumerate(folds)])
            # print(delta_score)
            i = np.argmin(delta_score)# for 
            fold_ids[i].append(sample_id)
        if verbose > 0:
            display([np.sum(f, axis = 0) for f in folds])
            print([np.sum(f) for f in folds])
            print([len(f) for f in fold_ids])
        return [(list(set(y.index) - set(f)), f) for f in fold_ids]

# def logit(y): return 1 / (1 + np.exp(-y))

def logit(y): return y
    
def add_feature(df, feat_cols):
    feat_cols = [f'{i}' for i in feat_cols]
    df['mean'] = np.mean(df[feat_cols].values, axis=-1)
    # df['min'] = np.min(df[feat_cols].values, axis=-1)
    # df['max'] = np.max(df[feat_cols].values, axis=-1)
    df['median'] = np.median(df[feat_cols].values, axis=-1)
    df['std'] = np.std(df[feat_cols].values.astype(float), axis=-1)
    # df['kurtosis'] = sss.kurtosis(df[feat_cols].values, axis=-1)
    # df['skew'] = sss.skew(df[feat_cols].values, axis=-1)
    # df['median_absolute_deviation'] = sss.median_absolute_deviation(df[feat_cols].values, axis=-1)
    # df['test'] = sss.tvar(df[feat_cols].values, axis=-1)
    return df

class CLR(sklearn.base.BaseEstimator):
    def __init__(self, reg = 1.0, l1_ratio = 0, tol = 1e-4):
        self.reg = reg
        self.l1_ratio = l1_ratio
        self.classes_ = np.array((0, 1))
        self.tol = tol
    
    def fit(self, X, y):
        wts = optimize(X.values, y.values, 
                           self.reg, self.l1_ratio, self.tol)
        self.wts = wts #/ max(wts.sum(), 1)# * 0.9
        # print(self.wts.sum())
        
    def predict(self, X):
        return logit((X * self.wts).sum(axis = 1)).values

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.stack(((1-preds), preds)).T
    


lr_params = {'reg': [  3e-5, 1e-4, 3e-4, 0.001, 0.003, 
                         0.1, 0.3, 1, 3],
             # 'intercept_scaling':[ 0.1, 1, 10],
                'l1_ratio': [0.0, 0.01, 0.03, 0.1, 0.3, 0.5,   
                            ],
                # 'tol': [1e-4, 3e-4, 7e-4, 5e-5]
            }


# %%time
rdfs = []# {target: [] for target in y.columns}
y_preds = {target: [] for target in y.columns}
y_test = {target: [] for target in y.columns}
models = {target: [] for target in y.columns}
all_wts = {target: [] for target in y.columns}
for target, pred_df in all_pred_dfs.items():
    print(target); 
    test_df = all_test_dfs[target]

    # print(test_df.head())
    # pred_df = logit(pred_df)# + np.random.normal(0, 0.1, size = pred_df.shape)#.iloc[:, :: 3]
    # pred_df = logit(pred_df)
    for bag_idx in range(10):
        n_folds = 5 # random.randrange(4, 6)
        folds_rs = datetime.datetime.now().microsecond
        folds = MultiLabelStratifiedKFold(n_folds, 
                             random_state = folds_rs,
                        full = random.choice([ True, False])
                                     ).split(
                        None, 
            y, 
            verbose = 0)


        for fold_idx, (train_idxs, test_idxs) in enumerate(folds):
            start = time.time()
            # inference_idxs = list( set(meta.sample_id) - set(train_idxs) ) 
            inference_idxs = test_idxs

            random.seed(datetime.datetime.now().microsecond)
                
            used_cols = random.sample(list(pred_df.columns),
            k = int( (0.3 + 0.3 * random.random())
                                       * len(pred_df.columns)) )
            pdf = pred_df.drop(columns =  used_cols)
            # pdf = add_feature(pdf, pdf.columns)
            
            test = test_df.drop(columns =  used_cols)
            # test = add_feature(test, test.columns)

            if 1:
                model = RandomizedSearchCV(
                    CLR(), lr_params,
                        cv = StratifiedKFold(
                                  n_splits = random.randrange(4, 6),
                                    # n_repeats = random.choice([1, 2]),
                                    shuffle = True,
                                    random_state = datetime.datetime.now().microsecond
                            ),
                                        n_iter = random.randrange(3, 4),#choice([3, 4]),
                                        n_jobs = -1,
                                scoring = 'neg_log_loss',
                                random_state = datetime.datetime.now().microsecond)

                model.fit(pdf.loc[train_idxs], 
                      y.loc[train_idxs, target])
                clf = model.best_estimator_

                rdf = pd.DataFrame(model.cv_results_
                        ).sort_values('rank_test_score').drop(
                                    columns = 'params')
                rdfs.append(rdf)

                y_pred = pd.Series(clf.predict(
                            pdf.loc[inference_idxs]),#[:, 1], 
                                       inference_idxs)
                # y_pred
                y_preds[target].append(y_pred)

                y_pred = pd.Series(clf.predict(
                            test),#[:, 1], 
                                       pred.index)
                # y_pred
                y_test[target].append(y_pred)

                # break;
                # if bag_idx == 0: print(clf)
                models[target].append(clf)
                all_wts[target].append(pd.Series(clf.wts, pdf.columns))

            else:
                # clf = BayesianRidge(n_iter=10, verbose=True)
                clf = LogisticRegression(penalty="l1", solver="liblinear", C=2)
                clf.fit(pdf.loc[train_idxs].values, y.loc[train_idxs, target].values)

                y_pred = pd.Series(clf.predict_proba(
                            pdf.loc[inference_idxs].values)[:, 1].reshape(-1), 
                                       inference_idxs)
                # y_pred
                y_preds[target].append(y_pred)

                y_pred = pd.Series(clf.predict_proba(
                            test.values)[:, 1].reshape(-1), 
                                       pred.index)
                # y_pred
                y_test[target].append(y_pred)
            
        # break;
    # break;
        # display(rdf)
    print()
    # if target >0:
    #     break


y_preds = {k: pd.concat(yp) for k, yp in y_preds.items()}
y_preds = {k: yp.groupby(yp.index).mean() for k, yp in y_preds.items()}

# print(y_preds)


metric = []
for i in range(9):
    metric.append(log_loss(gt[:,i], y_preds[i].values))
metric = np.mean(metric)
print(metric)


df = pd.read_csv('../data/raw/val_labels.csv')
df2 = df[['sample_id']]

y_test = {k: pd.concat(yp) for k, yp in y_test.items()}
y_test = {k: yp.groupby(yp.index).mean() for k, yp in y_test.items()}

sub_df = pd.read_csv('../data/raw/submission_format.csv')

if suf == '':
    sub_df = sub_df.head(312)
    metric = []
    for i in range(9):
        # df1 = df2.merge(y_test[i], on = ['sample_id'])
        # pred = df1.values
        pred = y_test[i].head(312).values
        metric.append(log_loss(df[target_cols[i]].values, pred))

        sub_df[target_cols[i]] = pred

    print(metric)
    metric = np.mean(metric)
    print(metric)
else:
    for i in range(9):
        pred = y_test[i].values
        sub_df[target_cols[i]] = pred

# print(sub_df.head())
sub_df.to_csv('../submissions/submission1.csv', index=False)