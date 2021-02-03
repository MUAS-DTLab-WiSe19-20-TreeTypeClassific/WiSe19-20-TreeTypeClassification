#!/bin/bash
echo "This is a simple shell script to start the pytorch implementation of pointnet++"
call source activate env_pytorch_p++
#cd /home/pflab/Desktop/Roessl/Pointnet++/Pytorch/Pointnet2_PyTorch-master/pointnet2/train
python -V
echo "Deactivate the venv"
call source deactivate
python -V
SOMEVAR="Var-txt"
echo "$SOMEVAR"
