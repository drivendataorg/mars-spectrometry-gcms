#source ./conda/setup_conda.sh
PROJET_NAME="mars2_2nd_place"
conda deactivate
conda remove -y --name $PROJET_NAME --all
conda create -y -n $PROJET_NAME python=3.8
conda activate $PROJET_NAME

pip3 install -r requirements.txt

pip install git+https://github.com/jacobgil/pytorch-grad-cam.git

