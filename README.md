


# WiSe19-20-TreeTypeClassification
============================


Deep Learning-based Classification of Tree Species and Standing Dead Trees on AWS Architecture

* Implemention of Pointnet2/Pointnet++ written in `PyTorch <http://pytorch.org>`_.


* Based on the implementation from `erikwijmans <https://github.com/erikwijmans/Pointnet2_PyTorch>`_.


The environment was customized locally and pushed to AWS via a docker image ``cevallos/pointnet2:dtlab``.


The custom ops used by Pointnet++ are currently **ONLY** supported on the GPU using CUDA.

Setup
-----

* Start an AWS Instance with GPU capabilities using the Ubuntu 18.04 Deep Learning AMI >= version 26 (the code is tested on versions 26 and 40)

* Pull this git to the home folder and create a separate results folder

  ::

    cd ~
    mkdir result
    git clone https://github.com/MUAS-DTLab-WiSe19-20-TreeTypeClassific/WiSe19-20-TreeTypeClassification.git

* Start custom image container with GPU capabilities and shared folders

  ::

    docker run -it --gpus all -v $HOME/WiSe19-20-TreeTypeClassification:$HOME/WiSe19-20-TreeTypeClassification -v $HOME/result:/pointnet2/result cevallos/pointnet2:dtlab bash
    
* Building `_ext` module

  ::

    cd /home/ubuntu/WiSe19-20-TreeTypeClassification/Pointnet++/Pytorch/Pointnet2_PyTorch-master/
    python setup.py build_ext --inplace


* To run the custom training, you also need to install this repo as a package

  ::

    pip install -e .


Example training
------------------

Two training examples are provided by ``pointnet2/train/train_sem_seg.py`` and ``pointnet2/train/train_cls.py``.
The datasets for both will be downloaded automatically by default.


They can be run via

::

  python -m pointnet2.train.train_cls

  python -m pointnet2.train.train_sem_seg


Both scripts will print training progress after every epoch to the command line.  Use the ``--visdom`` flag to
enable logging to visdom and more detailed logging of training progress.

Custom training
------------------

For this challenge a custom code for the tree data was implemented, it can be run via

::

  python ./pointnet2/train/train_cls_tschernobyl.py
