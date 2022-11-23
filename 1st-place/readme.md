# MARS SPECTROMETRY 2: GAS CHROMATOGRAPHY

Below you can find an outline of how to reproduce my solution for the `MARS SPECTROMETRY 2: GAS CHROMATOGRAPHY`.
If you run into any trouble with the setup/code or have any questions please contact me nghianguyenbkdn@gmail.com

## 1.INSTALLATION
- Ubuntu 18.04.5 LTS
- CUDA 11.2
- Python 3.7.5
- Training PC: 1x RTX3090 (or any GPU with at least 24Gb VRAM), 32GB RAM, at least 300Gb disk space.
- python packages are detailed separately in requirements.txt
```
$ conda create -n envs python=3.7.5
$ conda activate envs
$ pip install -r requirements.txt
$ pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## 2.DATA
* Download dataset and extract to `data/` folder.
* Folds split are provided as `train_folds.csv`, `val_folds.csv`, `train_folds_s2.csv`, `val_folds_s2.csv`. If you want to re-generate those files run `create_folds.py` (results can be different because of randomness). Others files are provided by drivendata.  
├── data    
│ ├── train_features   
│ ├── val_features   
│ ├── test_features  
│ ├── metadata.csv       
│ ├── submission_format.csv    
│ ├── train_folds.csv   
│ ├── train_folds_s2.csv   
│ ├── train_labels.csv  
│ ├── val_folds.csv       
│ ├── val_folds_s2.csv    
│ ├── val_labels.csv    

## 3. PREPROCESSING
* Download pre-processed data and model weights from [https://www.kaggle.com/datasets/nvnnghia/gcms-mars-weights] and extract to `code/` folder.  
  * The cache data can be reproduced by running `./preprocessing.sh` in `code/` folder (it takes about 7 hours).  
  * The model weights can be reproduced by executing the TRAINING STEP.   
* The `code/` folder now looks like  
├── code   
│ ├── configs    
│ ├── models    
│ ├── outputs   
│   ├── b5_bin005_sed_3ch_r6  
│   ├── b5_bin005_sed_3ch_r7  
│   ├── ...  
│ ├── cache   
│   ├── train_bin006.npy  
│   ├── train_bin005.npy  
│   ├── ...  
│ ├── train1.py   
│ ├── myswa.py   
│ ├── inference.py   
│ ├── cache_bin005.py   
│ ├── cache_bin005_s.py   
│ ├── cache_bin006.py   
│ ├── cache_bin006_s.py   
│ ├── blend_0308.py  
│ ├── ensemble.py  
│ ├── preprocessing.sh  
│ ├── inference.sh  
│ ├── train.sh  

## 4.INFERENCE
* Preprocessing step must be run before this step.
* Run the following command inside the `code/` folder.
```
$ ./inference.sh
```
   - The whole inference time takes around 3 hours.
   - It will produce 2 results files: `submission1.csv` and `submission2.csv`. `submission1.csv` is a blending of all models while `submission2.csv` is just average of all models. Both results file should yield a score around 0.144-0.146.

## 5.TRAINING
* Preprocessing step must be run before this step.
* To train all the models, run the following command inside the `code/` folder. 
```
$ ./train.sh
```
   - Due to the large number of models over 6 folds it takes around 10 days to train all models sequentially on 1x RTX3090. 
   - There are 10 models in total, if multiple gpus is available, each model can be trained separately on a single GPU for faster training.  
   - Each model is trained for several rounds with a set of commands, a set of commands within a model must be run sequentially. Refer to `code/train.sh` for detail.


