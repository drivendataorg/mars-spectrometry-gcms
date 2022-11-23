import numpy as np 
import pandas as pd 

target_cols = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']


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

print(len(paths))
suf = '_a'
# suf = ''

for i, path in enumerate(paths):
    if i == 0:
        pred = pd.read_csv(f'{path}/test{suf}.csv')
    else:
        if '///' not in path:
            pred1 = pd.read_csv(f'{path}/test{suf}.csv')
        else:
            print(path)
            pred1 = pd.read_csv(f'{path}/test{suf}_s2.csv')
        pred[target_cols] += pred1[target_cols]


pred[target_cols] /= len(paths)

print(pred.shape)
print(pred.head())
pred.to_csv('../submissions/submission2.csv', index=False)
