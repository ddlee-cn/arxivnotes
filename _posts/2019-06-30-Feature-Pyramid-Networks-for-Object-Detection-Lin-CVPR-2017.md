---
title: (FPN)Feature Pyramid Networks for Object Detection - Lin - CVPR 2017
tag:
- Object Detection
redirect_from: /Feature-Pyramid-Networks-for-Object-Detection-Lin-CVPR-2017.html
---



## Info

- Title: **Feature Pyramid Networks for Object Detection**
- Task: **Object Detection**
- Author: Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
- Date: March 2016
- Arxiv:  [1612.03144](https://arxiv.org/abs/1612.03144)
- Published: CVPR 2017

## Highlights
- Image pyramid to feature pyramid


<!-- more -->


## Motivation & Design
The understanding of picture information is often related to the modeling of position and scale invariance. In the more successful image classification model, the Max-Pooling operation models the invariance of position: the largest response is selected from the local, and the local position information of this response is ignored. In the direction of scale invariance, the convolution kernel (VGG) of different size receptive fields is added, and the large convolution kernel stack is used to feel a large range (GoogLeNet), and the structure of automatically selecting the size of the receptive field (Inception) also shows Its reasonable side.

As to the object detection task, unlike the classification task, the problem of detecting the size of the object is cross-category and in the same semantic scene.

An intuitive idea is to use different sizes of images to generate a feature map of the corresponding size, but this brings huge parameters, making the memory that can only run a single-digit image is not enough. Another idea is to directly use the feature map generated by different depth convolution layers, but the low-level features contained in the shallower feature map will interfere with the classification accuracy.

The method proposed in this paper is to pass the feature back down on the high-level feature map and construct the feature pyramid in reverse.

### Feature Pyramid Networks

![(FPN)Feature Pyramid Networks for Object Detection](https://i.imgur.com/S7aVS9v.png)

Starting from the picture, the cascading feature extraction is performed as usual, and a return path is added: starting from the highest feature map, the nearest neighbor is sampled down to get the return feature map of the same size as the low-level feature map. A lateral connection at the element position is then made to form features in this depth.

The belief in this operation is that the low-level feature map contains more location information, and the high-level feature map contains better classification information, combining the two to try to achieve the location classification dual requirements of the detection task.

## Performance & Ablation Study

The main experimental results of the article are as follows:

![(FPN)Feature Pyramid Networks for Object Detection](https://i.imgur.com/krnPtFI.png)

Comparing the different head parts, the input feature changes do improve the detection accuracy, and the lateral and top-down operations are also indispensable.

## Code
[Caffe2(FAIR's Detectron)](https://github.com/facebookresearch/Detectron)