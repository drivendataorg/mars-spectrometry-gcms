# 5th Place Winning Solution - Mars Spectrometry 2: Gas Chromatography

Username: jackson5

## Summary

My solution is an ensemble of ten identical Deep Neural Networks (DNNs) built using tensorflow. 
Each is trained using a combination of Label Distribution Learning (LDL) 
and a novel model averaging algorithm I name 'FixedSoup' inspired by ['GreedySoup'](https://arxiv.org/pdf/2203.05482.pdf) algorithm.
We construct the feature vectors using peak detection features alongside n-difference
features inspired by the derivatives produced by the [SG algorithm](https://pubs.acs.org/doi/10.1021/ac60214a047) in analytical chemistry. 

The final submission was produced by ensembling the models produced by `src/run_kfold_training.py`, which trains on both the `train` and `val` splits specified in `raw/metadata.csv`.
For convenience I have provided `src/run_single_training.py` as a simple use case, which only trains on the `train` split.

# Setup

0. Clone and change the directory
```
git clone https://github.com/Ninalgad/Mars-Spectrometry-Gas-Chromatography.git
cd Mars-Spectrometry-Gas-Chromatography
```

1. Create an environment using Python 3.8. The solution was originally run on Python 3.8.16. 
```
conda create --name ms2gs-submission python=3.8
```

then activate the environment
```
conda activate ms2gs-submission
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

(Optional) for GPU accelerated environments:

```
pip install tensorflow-gpu==2.9.2
```

3. Download the data from the competition page (and unzip) into `data/raw`

The structure of the directory before running training or inference should be:
```
Mars-Spectrometry-Gas-Chromatography
├── data
│   ├── processed      <- Output of training
│   └── raw            <- The original data files
│       ├── test_features
│       │   ├── S0809.csv
│       │   ├── S0810.csv
│       │   ...
│       ├── train_features
│       │   ├── S0000.csv
│       │   ...
│       ├── val_features
│       │   ├── S0809.csv
│       │   ...
│       ├── metadata.csv
│       ├── train_labels.csv
│       ├── val_labels.csv
│       └── submission_format.csv
├── models             <- Pre-trained model weights in h5 format
│   ├── s0.h5
│   ├── s1.h5
│   ...
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── make_dataset.py
│   ├── run_inference.py
│   ├── model.py
│   ├── loss.py
│   ├── run_kfold_training.py
│   ├── training_utils.py
│   └── run_single_training.py
├── README.md          <- The top-level README for developers using this project.
├── requirements.txt   <- The requirements file for reproducing the analysis environment
├── Makefile           <- Makefile with commands like `make requirements`
└── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
```

# Hardware

The solution was run on a Google colab notebook
- Number of CPUs: 4
- Processor: Intel(R) Xeon(R) CPU @ 2.20GHz
- Memory: 12 GB 
- GPU: Tesla T4

Both training and inference were run on GPU.
- Training time: ~ 6 hours
- Inference time: ~ 15 mins

# Run training

To run training using from the command line: `python src/run_kfold_training.py`. 

```
$ python src/run_kfold_training.py --help
Usage: run_kfold_training.py [OPTIONS]

Options:
  --model-dir PATH        Directory to save the
                          output model weights in
                          npy format  [default:
                          ./data/processed/]

  --features-dir PATH     Path to the raw features
                          [default: ./data/raw/]

  --labels-dir PATH       Path to the train_labels
                          csv and val_labels csv
                          [default: ./data/raw/]

  --metadata-path PATH    Path to the metadata csv
                          [default: ./data/raw/me
                          tadata.csv]

  --n-folds INTEGER       Number of folds, Must be
                          at least 2  [default:
                          10]

  --random-state INTEGER  Controls the randomness
                          of each fold  [default:
                          758625225]

  --debug / --no-debug    Run on a small subset of
                          the data and a two folds
                          for debugging  [default:
                          False]

  --help                  Show this message and
                          exit.
```

By default, trained model weights will be saved to `data/processed` in npy format. The model weights file that is saved is 11 MB.

To load the output weights use: `model.set_weights(np.load(your_weights.npy, allow_pickle=True))`

# Run inference

Trained model weights can be downloaded from this Google folder: https://drive.google.com/drive/folders/1ujIuxB5R62ik-5-5wg9gp5uvmR8qh57z?usp=sharing

Ensure the weights are located in the `models` folder.


To run inference from the command line: `python src/run_inference.py`

```
$ python src/run_inference.py --help
Usage: run_inference.py [OPTIONS]

Options:
  --model-dir PATH               Directory of the
                                 saved model
                                 weights
                                 [default:
                                 ./models/]

  --features-path PATH           Path to the raw
                                 features
                                 [default:
                                 ./data/raw/]

  --submission-save-path PATH    Path to save the
                                 generated
                                 submission
                                 [default: ./data
                                 /processed/submis
                                 sion.csv]

  --submission-format-path PATH  Path to save the
                                 submission format
                                 csv  [default: .
                                 /data/raw/submiss
                                 ion_format.csv]

  --metadata-path PATH           Path to the
                                 metadata csv
                                 [default: ./data
                                 /raw/metadata.csv
                                 ]

  --debug / --no-debug           Run on a small
                                 subset of the
                                 data and a single
                                 model for
                                 debugging
                                 [default: False]

  --help                         Show this message
                                 and exit.
```

By default, predictions will be saved out to `data/processed/submission.csv`.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>