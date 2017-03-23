# Homographynet
This is an implementation of the paper [Deep Image Homography Estimation](https://arxiv.org/pdf/1606.03798.pdf) with tensorflow

## dataset
ms-coco  
> train2014: 82783 images  
> val2014: 40504 images  
> test2014: 40775 images  

## train_mycnn.py
use the cnn network by myself for train  

## main.py
use existing cnn framework for training, dropout & bn the matter  

## train_2.py
use data_generation.py to generate images first. Then start train
