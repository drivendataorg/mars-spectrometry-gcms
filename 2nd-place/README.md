# Mars Spectrometry 2: Gas Chromatography - the 2nd place solution
Username: `dmitryakonovalov`
<br>
**License**: MIT

## Summary

My solution evolved from the [1st place solution](https://github.com/drivendataorg/mars-spectrometry/tree/main/1st%20Place) from 
the previous  [Mars Spectrometry: Detect Evidence for Past Habitability](https://www.drivendata.org/competitions/93/nasa-mars-spectrometry/).

[//]: # (- What kind of data preprocessing did you do?)
Each sample csv-file was converted to a 2D representation, where 256 rounded integer mass values (y-axis) 
and 192 rescaled time slots (x-axis) were used. 
The sample intensities at each mass (y-axis) and 
time (x-axis) bin values were added without any further preprocessing.


[//]: # (- What types of models and/or pretrained model backbones did you use? Is your final prediction an ensemble?)
The following pretrained backbones from [timm](https://github.com/rwightman/pytorch-image-models) were used 
(run-config, timm-backbone, random-seed, metric):  
```
cfg.MODELS = [('stage3a/302val_cls3_normM', 'hrnet_w64', '3', 'loss_val'),
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
```
Two variations of classification heads were used in the final models. 
First, `_cls3_`-tag, was the standard timm head followed by the required 9-class classifier linear layer. 
The second head (tagged `_flatM_`) retained mass-dimension 
(after the backbone's features, only time-dimension was averaged) then followed by the 9-class classifier.  

Five-fold cross-validation was used to train five versions of each model. 
When training, on-GPU batch-wise augmentations were performed by resizing the time dimension (x-axis). 
Some run configurations used mixup augmentations (tagged by `_mix01` in the above list). 
For predictions, TTA (test-time-augmentations) was performed by averaging model outputs from five different time-sizes 
(5 steps of 32, centred at 192). For the submission, the outputs of all models were ensembled by averaging clipped prediction logits
and then applying the sigmoid function. 

Two step-wise ensemble algorithms were used. First was "grow-by-one", which started 
with the lowest-oof-loss model and progressively 
added a model which, in combination with already selected models, reduced the oof-loss to a new minimum value.
The second was "remove-by-one" 
(as per original from [1st place solution](https://github.com/drivendataorg/mars-spectrometry/tree/main/1st%20Place)  ), 
which started with all calculated models and progressively removed one model 
so that the remaining models yielded the best possible oof-loss.
 

# Setup

Tested on Ubuntu 20.04 with CUDA 10.2 (e.g. `nvidia-smi` and `nvcc --version` are working) and `python=3.8`, `torch==1.11.0`.

Install via conda/miniconda (`mars2_2nd_place` conda env will be created):
```
source setup_conda.sh
```
Or 
```
pip3 install -r requirements.txt
```

Data preparation:
----------------

The train/val and test datasets are assumed to be in `~/dev/drivendata_mars` directory in the following format/locations:
```
    ~/dev/drivendata_mars
        /test_features
        /train_features
        /val_features
        /metadata.csv
        /train_labels.csv
        /val_labels.csv
```
Modify the `cfg.DATA_DIR = expanduser('~/dev/drivendata_mars')` in `config_main_v4.py` to specify a different location.

# Hardware
The solution was run on NVIDIA GeForce GTX 1080 Ti and 2080 

Training time: Final models (each with 5 folds) took less than 30 hours (less than 0.5 hours per model/fold) 
on one NVIDIA 1080Ti GPU.

Inference time: less than one hour for the given test+val data.  


# Run training

Five-fold cross-validation was used, see  `folds_n5_seed42_v1a.csv`. To re-generate `folds_n5_seed42_v1a.csv`, run 
```
python preprocess_data_v3b.py
```

To train (to experiment/debug) one run-config (see examples in `stage3a`-folder), timm-model-name, one fold and one random-seed(fixed), run 
```
python train_v4.py  <see available args inside code>
```
To train all folds for one or more run-configs (see `stage3a`-folder), run 
```
python train_all_v4.py  <see available args inside code>
```
which is currently set up (via `cfg.MODELS = [...]` in `config_main_v4.py`) to rerun all models (5 folds each) used in the final submission. 

An example of running all/any models via bash scripts is here: `bash_scripts/train_example.sh`

All outputs are written to `'~/tmp/ddm'`, which can be changed via `cfg.OUTPUT_DIR = expanduser('~/tmp/ddm')` in `config_main_v4.py` or via `--output_dir` parameter.

The first time a sample is requested from the dataset (see `dataset_v4.py`), 
it will be converted to an image and saved to a temporary directory 
(`~/tmp/ddm`, set via `cfg.TMP_DIR = expanduser('~/tmp/ddm')` 
in `config_main_v4.py`). So, the very first epoch of the very first run-config will be slow.

The final models require less than 30GB of disk space.
All per-model-fold TTA predictions and pytorch weights are saved in the output directory set 
via `--output_dir=<your dir destination>` 
command line argument.
Note that for training, an internet connection is required as timm pretrained model weights need to be automatically downloaded.


# Run inference

Original submission with oof loss=0.08631 (without model weights) is 
here: `/test_results_saved/ddm_v4_outputs.zip`

Note that all per-model-fold predictions are automatically calculated at the end of the training phase. 
If required for new test data, 
download models' weights with oof loss=0.085937 (should be better than the original submission) 
from here (NOTE only available until the end of 2023): 
[https://cloudstor.aarnet.edu.au/plus/s/AOCiwmQ0pzvHwJG](https://cloudstor.aarnet.edu.au/plus/s/AOCiwmQ0pzvHwJG).
Unzip into e.g. `"~/tmp/ddm_2nd_place_weights"`.
Then run 
```
predict_all_v4.py --output_dir "~/tmp/ddm_2nd_place_weights" 
```
It takes about one-two minute per model fold. With the current setup of 13 models, it should take 
about one hour to recalculate predictions for the competition's test+val datasets.  


# Generate submission by ensembling all models
To generate the original submission:
Unzip all per-model predictions (see `test_results_saved/ddm_v4_outputs.zip`) into `~/tmp/ddm_v4_outputs` 
and run
```
python test_v4.py --results_dir "~/tmp/ddm_v4_outputs"
```
It will save submission files into `~/tmp/ddm_v4_outputs_test`  and `./test_results`.
The original submission should be regenerated as 
`./test_results/oof0.086313_ddm_v4_outputs_test_mean_logits_clip1e4.csv`,  
which should be the same as my 2nd place submission (see `test_results_saved/ddm_v4_predictions.zip`)


To ensemble a new submission (before or after running predict_all_v4.py), run
```
python test_v4.py --results_dir "~/tmp/ddm_2nd_place_weights" 
```


