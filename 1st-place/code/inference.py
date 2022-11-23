import os 

cfgs = [
    f'b5_bin005_sed_3ch_r6', #0.1520
    'b5_bin005_sed_3ch_r7', #0.1558

    'b6_bin005_sed_3ch_r6', #0.1456
    'b6_bin005_sed_3ch_r7', #0.1474

    'b7_bin005_sed_3ch_r6', #0.1505
    'b7_bin005_sed_3ch_r7', #0.1476

    f'm13_bin005_r9', #0.1435
    f'm13_bin005_r10', #0.1444
    f'm13_bin006_r6', #0.1444
    f'm13_bin006_r7', #0.1463
    f'm13_bin006_r8', #0.1465

    f'm10_bin005_v1_r5', #0.1517
    f'm10_bin005_v1_r6', #0.1513
    f'm10_bin005_v1_r7', #0.1509

    f'b5_bin005_sed_3ch_v1_r6',
    f'b5_bin005_sed_3ch_v1_r7',

    f'b5_bin005_sed_3ch_v1_r5/',
    f'b5_bin005_sed_3ch_v1_r6/',
    f'b5_bin005_sed_3ch_v1_r7/',
    'm13_bin006_r7/', #0.1458
    f'm13_bin006_r8/', #0.1446

    'm13_1_bin006_r6',
    f'm13_1_bin006_r7',
    f'm13_1_bin006_r8',
]

for cfg in cfgs:
    if '/' in cfg:
        cfg = cfg.replace('/','')
        os.system(f'CUDA_VISIBLE_DEVICES=0 python train1.py -C {cfg} -S 2 -A 1 -M test') #inference on val and test set
        os.system(f'CUDA_VISIBLE_DEVICES=0 python train1.py -C {cfg} -S 2 -A 1 -M val') #create oof file
    else:
        os.system(f'CUDA_VISIBLE_DEVICES=0 python train1.py -C {cfg} -A 1 -M test') #inference on val and test set
        os.system(f'CUDA_VISIBLE_DEVICES=0 python train1.py -C {cfg} -A 1 -M val') #create oof file

    # break