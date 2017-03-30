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
![1](http://i1.piimg.com/567571/490315a068fd15bd.png)  

my result Vs. the author's result on the same image  
![2](http://i1.piimg.com/567571/e822ab2e728f381b.png)  
![2_2](http://i4.buimg.com/567571/3b9f54d83c67248a.png)  

bad example:  
![3](http://i4.buimg.com/567571/3904cd85018bcff9.png)
