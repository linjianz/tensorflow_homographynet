# Homographynet
This is an implementation of the paper [Deep Image Homography Estimation](https://arxiv.org/pdf/1606.03798.pdf) with tensorflow

## dataset
ms-coco  
> train2014: 82783 images  
> val2014: 40504 images  
> test2014: 40775 images  

## train_mycnn.py
use the cnn network by myself for train, no need to pre-save the generated data  

## test_mycnn.py
test only one image a time, output the four-pair offsets predicted from HomographyNet. Then use data_process.m to visulize the results.  

## train_net.py
use existing cnn framework for training, dropout & bn have something wrong  

## data_generation.py
if you like, just generate the training data
