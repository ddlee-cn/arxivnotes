---
title: "DSOD: learning deeply supervised object detectors from scratch - Shen - ICCV 2017 - Caffe Code"
tag:
- Object Detection
redirect_from: /DSOD-learning-deeply-supervised-object-detectors-from-scratch-Shen-ICCV-2017.html
---



## Info
- Title: **DSOD: learning deeply supervised object detectors from scratch**
- Task: **Object Detection**
- Author: Z. Shen, Z. Liu, J. Li, Y. Jiang, Y. Chen, and X. Xue
- Date: Aug. 2017
- Arxiv: [1708.01241](https://arxiv.org/abs/1708.01241)
- Published: ICCV 2017

## Highlights & Drawbacks
- Object Detection without pre-training
- DenseNet-like network

<!-- more -->


## Motivation & Design

A common practice that used in earlier works such as R-CNN is to pre-train a backbone network on a categorical dataset like ImageNet, and then use these pre-trained weights as initialization of detection model. Although I have once successfully trained a small detection network from random initialization on a large dataset, there are few models trained from scratch when the number of instances in a dataset is limited like Pascal VOC and COCO. Actually, using better pre-trained weights is one of the tricks in detection challenges. DSOD attempts to train the detection network from scratch with the help of "Deep Supervision" from DenseNet.

The 4 principles authors argued for object detection networks:

    1. Proposal-free
    2. Deep supervision
    3. Stem Block
    4. Dense Prediction Structure

![DSOD: learning deeply supervised object detectors from scratch](https://i.imgur.com/amvcbcK.png)


## Performance & Ablation Study

 DSOD outperforms detectors with pre-trained weights.
![DSOD: learning deeply supervised object detectors from scratch](https://i.imgur.com/1dt4lad.png)

Ablation Study on parts:
![DSOD: learning deeply supervised object detectors from scratch](https://i.imgur.com/vKRUrAf.png)



## Code

[Caffe](https://github.com/szq0214/DSOD)