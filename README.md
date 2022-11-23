[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/nasa-mars-curiosity.jpg)](https://mars.drivendata.org/)

# Mars Spectrometry 2: Gas Chromatography

## Goal of the Competition
In this challenge, the competitors' goal was to build a model to automatically analyze mass spectrometry data collected for Mars exploration in order to help scientists in their analysis of understanding the past habitability of Mars.

Their models detect the presence of certain families of chemical compounds in data collected from performing an analytical technique called gas chromatography-mass spectrometry (GCMS) on a set of analog samples. The winning techniques seen in this repo may be used to help analyze data from Mars, and potentially even inform future designs for planetary mission instruments performing in-situ analysis. 

## What's in this Repository

This repository contains code from winning competitors in the [Mars Spectrometry 2: Gas Chromatography](https://www.drivendata.org/competitions/97/nasa-mars-gcms/) DrivenData challenge. It is the companion to the first Mars spectrometry challenge ([Mars Spectrometry: Detect Evidence for Past Habitability](https://www.drivendata.org/competitions/93/nasa-mars-spectrometry/)), which was conducted using evolved gas analysis (EGA) data.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | User | Private Score | Summary of Model
--- | --- | ---   | ---
1  | [nvnn](https://www.drivendata.org/users/nvnn/) | 0.144 | Ensemble of a 1D CNN-transformer that is fed into a 1D event detection network and a 2D CNN (each channel has different preprocessing method), used pre-trained CNN backbones.
2   | [dmitryakonovalov](https://www.drivendata.org/users/dmitryakonovalov/) | 0.149 | Ensemble of 13 2D CNNs with different preprocessing methods (including variable time bins), used pre-trained backbones.
3   | [ouranos](https://www.drivendata.org/users/ouranos/) | 0.150 | Ensemble of 4 models: logistic regression, ridge classification (with feature selection), simple CNN, efficientnet CNN, used pre-trained backbones and statistical features.
Bonus Prize | [jackson5](https://www.drivendata.org/users/jackson5/) | 0.151 | Averaged weights of 10 deep learning models, engineered features to describe peaks 

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: [Mars Spectrometry 2: Gas Chromatography](https://www.drivendata.co/blog/mars-spectrometry-gcms-benchmark)**
