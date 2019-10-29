---
title: (RetinaNet)Focal loss for dense object detection - Lin  - ICCV 2017
tag:
- Object Detection
redirect_from: /RetinaNet-Focal-loss-for-dense-object-detection-Lin-ICCV-2017.html
---

## Info

- Title: **Focal loss for dense object detection**
- Task: **Object Detection**
- Author: T. Lin, P. Goyal, R. B. Girshick, K. He, and P. Dollár
- Date: Aug. 2017
- Arxiv: [1708.02002](https://arxiv.org/abs/1708.02002)
- Published: ICCV 2017(Best Student Paper)

## Highlights & Drawbacks
- Loss function improvement
- For *Dense* samples from single-stage models like SSD

<!-- more -->

## Motivation & Design


In single-stage models, a massive number of training samples are calculated in the loss function at the same time, because of the lack of proposing candidate regions. Based on findings that the loss of a single-stage model is dominated by easy samples(usually backgrounds), Focal Loss introduces a suppression factor on losses from these easy samples, in order to let hard cases play a bigger role in the training process. 


![(RetinaNet)Focal loss for dense object detection](https://i.imgur.com/C6uuJrQ.png)

Utilizing focal loss term, a dense detector called RetinaNet is designed based on ResNet and FPN:

![(RetinaNet)Focal loss for dense object detection](https://i.imgur.com/62SFpNT.png)


## Performance & Ablation Study
The author’s experiments show that a single-stage detector can achieve comparable accuracy like two-stage models thanks to the proposed loss function. However, Focal Loss function brings in two additional hyper-parameters. The authors use a grid search to optimize these two hyper-parameters, which is not inspiring at all since it provides little experience when using the proposed loss function on other datasets or scenarios. Focal Loss optimizes the weight between easy and hard training samples in the loss function from the perspective of sample imbalance.

![(RetinaNet)Focal loss for dense object detection](https://i.imgur.com/IObdCS1.png)


## Code

[Caffe2(FAIR's Detectron)](https://github.com/facebookresearch/Detectron)