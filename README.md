# EIRM
EIRM-Net:A Lightweight Deep Learning Framework with EIRM for Cross-Domain Few-Shot Learning
## Preparation
1. Change the ROOT_PATH value in the following files to yours:
    - `datasets/mini_imagenet.py`
    - `datasets/tiered_imagenet.py`
    - `datasets/cifarfs.py`

2. Download the datasets and put them into corresponding folders that mentioned in the ROOT_PATH:<br/>
    - ***mini*ImageNet**: download from [CSS](https://github.com/anyuexuan/CSS) and put in `data/mini-imagenet` folder.

    - ***tiered*ImageNet**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/tiered-imagenet` folder.

    - **CIFARFS**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/cifar-fs` folder.
## Pre-trained Models
Put the content in the save folder. To evaluate the model, run the test.py file with the proper save path as in the next section.
## Experiments
To train on 1-shot and 5-shot MiniImageNet:<br/>

python train_stage1.py --dataset mini --train-way 50 --train-batch 100 --save-path ./save/mini-stage1

python train_stage2.py --dataset mini --shot 1 --save-path ./save/mini-stage2-1s --stage1-path ./save/cifarfs-stage1 --train-way 20

bash run_training_with_logging_cifarfs_5.sh

To evaluate on 5-way 1-shot and 5-way 5-shot CIFAR-FS:<br/>

python test.py --dataset mini --shot 1 --save-path ./save/mini-stage3-1s
