# Homographynet
This is an implementation of the paper [Deep Image Homography Estimation](https://arxiv.org/pdf/1606.03798.pdf) with tensorflow

## Dataset
ms-coco 

| dataset | image numbers |
|:---:|:---:|
|train2014|82783|
|val2014|40504|
|test2014|40775|

## Program list
### train_mycnn.py
Build the network according to the paper completely.  
No need to pre-save the generated data, the program genetated the image pairs automatically.
### test_mycnn.py
Test only one image a time, output the four-pair offsets predicted from HomographyNet.  
Then use data_process.m to visulize the results.
### train_net.py
Use existing cnn framework for training, something wrong ~v~.
### data_generation.py
If you like, just generate the training data.
## Result
I test 200 images on test2014, Mean Corner Error = 12.6578 (image size is 320x240).  
The original thesis is 9.2 (image size is 640x480). But I believe my result could be better.  

good example:  
![](http://ogmp8tdqb.bkt.clouddn.com//18-1-8/30625113.jpg?imageView2/2/h/200/interlace/0/q/100)  

my result Vs. the author's result on the same image  
![](http://ogmp8tdqb.bkt.clouddn.com//18-1-8/40994198.jpg?imageView2/2/h/200/interlace/0/q/100)  

![](http://ogmp8tdqb.bkt.clouddn.com//18-1-8/43294698.jpg?imageView2/2/h/200/interlace/0/q/100)  

bad example:  
![](http://ogmp8tdqb.bkt.clouddn.com//18-1-8/91429979.jpg?imageView2/2/h/200/interlace/0/q/100)  
