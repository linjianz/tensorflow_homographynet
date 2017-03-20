# Homographynet
This is an implementation of the paper [Deep Image Homography Estimation](https://arxiv.org/pdf/1606.03798.pdf) with tensorflow

## dataset
ms-coco  
> train2014: 82783 images  
> val2014: 40504 images  
> test2014: 40775 images  

## main_train.py
for each image, generate 8 pairs of samples

## main_test.py
for each image, test the loss of generated sample

## train_2.py
use data_generation.py to generate images first. Then start train
