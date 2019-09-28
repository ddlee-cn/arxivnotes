---
title: "R-FCN: Object Detection via Region-based Fully Convolutional Networks - Dai - NIPS 2016 - MXNet Code"
tag:
- Object Detection
---



## Info
- Title: **R-FCN: Object Detection via Region-based Fully Convolutional Networks**
- Task: **Object Detection**
- Author: Jifeng Dai, Yi Li, Kaiming He, and Jian Sun
- Arxiv: [1605.06409](https://arxiv.org/abs/1605.06409)
- Published: NIPS 2016


## Highlights
- Full convolutional network, sharing weights across ROIs


## Motivation & Design
The article points out that there is an unnatural design of the framework before the detection task, that is, the feature extraction part of the full convolution + the fully connected classifier, and the best performing image classifier is a full convolution structure (ResNet, etc.). One point is caused by the contradiction between the translation invariance of the classification task and the translation sensitivity of the detection task. In other words, the detection model uses the feature extractor of the classification model, and the position information is lost. This article proposes to solve this problem by using a "location-sensitive score map" approach.

### Position-sensitive score maps & Position-sensitive RoI Pooling

There are two important operations for generating a position sensitive score map. One is to generate a more "thick" feature map, and the other is to selectively enter a feature map during RoI Pooling.

![R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://i.imgur.com/Nq92vNC.png)


In Faster R-CNN, RoI is obtained through RPN, converted into classification task, and a certain amount of convolution operation (conv5 part in ResNet) is added, and this partial convolution operation cannot be shared. R-FCN focuses on the full convolution structure, using the freedom of the convolution operation in the dimension of Channel, giving its position sensitive meaning. Here are the specific operations:

- In the last layer of the full convolutional network, generate feature maps of k^2(C+1) Channels, where C is the number of categories, and k^2 represents the k×k grid, which is used to detect the k of the target object, respectively. ×k parts. That is, the feature maps of different channels represent different parts of the object (such as the upper left part and the lower right part).
- After mapping the Proposal obtained by the RPN network to the feature map obtained in the previous step (thickness is k×k×(C+1),), correspondingly, the RoI is divided into k×k bins, for the first (i, j) bin, considering only the (C+1) feature maps corresponding to the (i, j) position, perform the following calculation: where (x0, y0) is the anchor of this RoI, and the result is (i, j) No. bin's corresponding score for the C category.
- After the previous step, the result obtained by each RoI is the fractional tensor of k^2(C+1) size, k×k encodes the partial score information of the object, and after the vote (average), the (C+1) dimension is obtained. The score vector, then access softmax to get the probability of each class.

In the second step above, "selecting only the feature map (i, j)" is the key to the meaning of the location information.

![R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://i.imgur.com/k9u9Cxw.png)

In the network structure designed in this way, all the learnable parameters are distributed in the shareable convolution layer, thus improving the training and test performance.

## Performance & Ablation Study

The comparison with Faster R-CNN shows that R-FCN achieves better accuracy while maintaining shorter inference time.
![R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://i.imgur.com/6DYmf4m.png)

## Code
[MXNet](https://github.com/daijifeng001/R-FCN)



## Related
- [You Only Look Once: Unified, Real Time Object Detection - Redmon et al. - CVPR 2016](https://arxivnote.ddlee.cn/You-Only-Look-Once-Unified-Real-Time-Object-Detection-Redmon-CVPR-2016.html)
- [An analysis of scale invariance in object detection - SNIP - Singh - CVPR 2018](https://arxivnote.ddlee.cn/An-analysis-of-scale-invariance-in-object-detection-SNIP-Singh-CVPR-2018.html)
- [Faster R-CNN: Towards Real Time Object Detection with Region Proposal - Ren - NIPS 2015](https://arxivnote.ddlee.cn/Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Ren-NIPS-2015.html)
- [(FPN)Feature Pyramid Networks for Object Detection - Lin - CVPR 2017](https://arxivnote.ddlee.cn/Feature-Pyramid-Networks-for-Object-Detection-Lin-CVPR-2017.html)
- [(RetinaNet)Focal loss for dense object detection - Lin - ICCV 2017](https://arxivnote.ddlee.cn/RetinaNet-Focal-loss-for-dense-object-detection-Lin-ICCV-2017.html)
- [YOLO9000: Better, Faster, Stronger - Redmon et al. - 2016](https://arxivnote.ddlee.cn/YOLO9000-Better-Faster-Stronger-Redmon-2016.html)