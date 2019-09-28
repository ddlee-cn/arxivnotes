---
title: (DenseNet)Densely Connected Convolutional Networks - Huang - CVPR 2017
tag:
- Image Classification
- CNN
redirect_from: /Densely-Connected-Convolutional-Networks-Huang-CVPR-2017.html
---

## Info
- Title: **Densely Connected Convolutional Network**
- Task: **Image Classification**
- Author: Gao Huang, Zhuang Liu, Laurens van der Maaten and Kilian Weinberger 
- Arxiv: [1608.06993](https://arxiv.org/abs/1608.06993)
- Published: CVPR 2017(Best Paper Award)

## Highlights

DenseNet takes the idea of ​​shortcut-connection to its fullest. Inside a DenseBlock, the output of each layer is created with the following layers. It is important to note that unlike the addition in ResNet, the DenseNet connection shortcut is Concat, so the deeper the layer, the more the input channel number. Big.


<!-- more -->

## Motivation & Design

![(DenseNet)Densely Connected Convolutional Networks](https://i.imgur.com/WTz22Su.png)

The entire network is divided into Dense Block and Transition Layer. The former is densely connected internally and maintains the same size feature map. The latter is the connection layer between DenseBlocks and performs the downsampling operation.

Within each DenseBlock, the accepted data dimension will become larger as the number of layers deepens (because the output of the previous layer is spliced ​​continuously), and the rate of growth is the initial channel number. The article calls the channel number as the growth rate. A hyper-parameter of the model. When the initial growth rate is 32, the number of channels in the last layer will increase to 1024 under the DenseNet121 architecture.

[Netscope Visualization] (http://ethereon.github.io/netscope/#/gist/56cb18697f42eb0374d933446f45b151) and source code: [awesome_cnn](https://github.com/ddlee96/awesome_cnn).

## Performance & Ablation Study

The authors have done experiments on both CIFAR and ImageNet. DenseNet has achieved comparable performance with ResNet. After adding Botleneck and a part of the compression technique, it can achieve the same effect as ResNet with fewer parameters:

![(DenseNet)Densely Connected Convolutional Networks](https://i.imgur.com/s58rsYr.png)

## Code
[Caffe](https://github.com/liuzhuang13/DenseNet)