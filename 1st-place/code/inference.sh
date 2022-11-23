#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py 

python blend_0308.py

python ensemble.py
