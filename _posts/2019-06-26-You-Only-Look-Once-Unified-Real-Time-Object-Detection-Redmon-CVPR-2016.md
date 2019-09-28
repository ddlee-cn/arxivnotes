---
title: "You Only Look Once: Unified, Real Time Object Detection - Redmon et al. - CVPR 2016"
tag:
- Object Detection
- Middle-Level Vision

---

## Info
- Title: **You Only Look Once: Unified, Real Time Object Detection**
- Task: **Object Detection**
- Author: J. Redmon, S. Divvala, R. Girshick, and A. Farhadi
- Arxiv: https://arxiv.org/abs/1506.02640
- Date: June. 2015
- Published: CVPR 2016

## Highlights & Drawbacks
- Fast.
- Global processing makes background errors relatively small compared to local (regional) based methods such as Fast RCNN.
- Generalization performance is good, YOLO performs well when testing on art works.
- The idea of YOLO meshing is still relatively rough, and the number of boxes generated by each mesh also limits its detection of small objects and similar objects.


<!-- more -->

## Motivation & Design

![You Only Look Once: Unified, Real Time Object Detection](https://i.imgur.com/ZO5EiVs.png)


### Procedure
1. Prepare the data: Scale the image and divide it into equal-divided grids. Each grid is assigned to the sample to be predicted by the ground truth IOU.
2. Convolutional network: changed by GoogLeNet, each grid predicts a conditional probability value for each category, and generates B boxes on a grid basis, each box predicts five regression values, four representation locations The fifth characterizes the probability and location of the object containing the object (note that it is not a certain type of object) (represented by the IOU). When testing, the score is calculated as follows:

$$
\operatorname{Pr}\left(\text { Class }_{i} | \text { Object }\right) * \operatorname{Pr}(\text { Object }) * \operatorname{IOU}_{\text { ped }}^{\text { truth }}=\operatorname{Pr}\left(\text { Class }_{i}\right) * \operatorname{IOU}_{\text { pred }}^{\text { truth }}
$$
    The first item on the left side of the equation is predicted by the grid, and the last two items are predicted by each box, and the combination becomes a score that each box contains objects of different categories.
    Therefore, the number of prediction values outputted by the convolution network is S×S×(B×5+C), S is the number of grids, B is the number of boxes generated for each grid, and C is the number of categories.
    3. Post-processing: Box obtained by filtering with NMS

### Loss

![You Only Look Once: Unified, Real Time Object Detection](https://i.imgur.com/dBqrPc5.png)



The loss function is divided into three parts: coordinate error, object error, and class error. In order to balance the effects of category imbalance and large and small objects, weights are added to the loss and the root length is taken.

## Performance & Ablation Study

![You Only Look Once: Unified, Real Time Object Detection](https://i.imgur.com/RJQH4lU.png)




![You Only Look Once: Unified, Real Time Object Detection](https://i.imgur.com/Bx7fLGT.png)

Compared to Fast-RCNN, YOLO's background false detections account for a small proportion of errors, while position errors account for a large proportion (no log coding).

## Code
- [Project Site(Contains newest v3)](https://pjreddie.com/darknet/yolo/)
- [Darknet](https://github.com/pjreddie/darknet)