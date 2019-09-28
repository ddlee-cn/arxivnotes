---
title: Speed/accuracy trade-offs for modern convolutional object detectors - Huang et al. - CVPR 2017 - TensorFlow Code
tag:
- Object Detection
redirect_from: /Speed-accuracy-trade-offs-for-modern-convolutional-object-detectors-Huang-CVPR-2017.html
---

## Info
- Title: **Speed/accuracy trade-offs for modern convolutional object detectors**
- Task: **Object Detection**
- Author: Jonathan Huang, Vivek Rathod, et al.
- Arxiv: [1611.10012](https://arxiv.org/abs/1611.10012)
- Date: Nov. 2016
- Published: CVPR 2017


## Highlights & Drawbacks
- This article is an empirical study.  The feature extraction network is separated as a component of the meta-structure, and the Proposal number, the input image size, and the feature map are generated. 
- SOTA is achieved via hyper-parameter optimization


<!-- more -->


## Motivation & Design
Several parallel experiments are conducted to explore trade-off in terms of accuracy and speed.

Three meta-structures: Faster-RCNN, R-FCN and SSD.
![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/nY75KG8.png)


### Performance & Ablation Study

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/X60egSq.png)

- The horizontal and vertical dimensions represent speed and accuracy respectively. The more the horizontal axis is to the left, the less time is used. The higher the vertical axis is, the better the mAP performance is. Therefore, the sweet spot should be distributed in the upper left corner.
- Two super dimensions are meta-structures and feature extraction networks, meta-structures are represented by shapes, and feature extraction networks are represented by colors
- The dotted line represents the ideal trade-off boundary

analysis:

- The highest accuracy is extracted by the Faster-RCNN meta-structure, Inception-ResNet, high-resolution images, using a larger feature map, as shown in the upper right corner of the figure.
- The fastest performance in faster networks is achieved by SSDs using Inception and Mobilenet
- The sweet spot zone feature extraction network is ruled by ResNet, and the less Proposal Faster-RCNN can be comparable to R-FCN
- Feature extraction network, Inception V2 and MobileNet in high speed zone, Incep-ResNet and ResNet in sweet spot and high precision zone, Inception V3 and VGG away from ideal boundary (dashed line)

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/4VeN6bo.png)

The figure above shows the influence of the feature extraction network on the three meta-structures. The horizontal axis is the classification accuracy of the feature extraction network, and the vertical axis is the mAP performance on the detection task. It can be seen that the SSD has the smallest difference in the vertical axis direction, and the Faster - RCNN and R-FCN are more sensitive to feature extraction networks.

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/poFNPP4.png)

The horizontal axis of the above figure is a different feature extraction network. The group is a comparison of three meta-structures, and the vertical axis is the mAP of objects of different sizes.

It can be seen that in the detection of large objects, when using a smaller network, the effect of SSD is comparable to that of the two-stage method, and the deeper feature extraction network improves the detection of medium-sized and small objects in the two-stage method (ResNet101 and Incep-ResNet has shown a two-stage approach to lifting on small objects)

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/uVs0YKz.png)


The figure above shows the effect of the input picture size on the mAP. Pictures with high scores help with small object detection and therefore have higher precision, but the relative speed will be slower.

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/lif2wfc.png)


The above diagram explores the effect of Proposal numbers in a two-stage approach. The left side is Faster-RCNN, the right side is R-FCN, the solid line is mAP, and the dashed line is the inferred time.

analysis:

- Compared to R-FCN, Faster-RCNN infers time is quite sensitive to the number of Proposal (because there is per ROI calculation)
- Reduce the number of Proposal without causing a fatal drop in accuracy

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/5wyxEjW.png)

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/hEKxVIV.png)

The above two figures are records of FLOPS, which is more neutral than GPU time. In Figure 8, the GPU part shows the difference between ResNet and Inception (about 45 degrees line, FLOPS is equivalent to GPU time), the article thinks that the decomposition operation ( Factorization) reduces FLOPs, but increases memory IO time, or the GPU instruction set is more suitable for dense convolution calculations.

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/vAmf2Uo.png)

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/tNZXLGA.png)

The above two figures are an analysis of memory usage. In general, the more streamlined the feature extraction network, the smaller the feature map size, the less memory is used, and the running time is shorter.

Finally, the article describes their ensemble ideas, in a series of different strike, loss and configuration of Faster-RCNN (ResNet and Incep-ResNet for feature extraction network), greedily choose the AP on the verification set is higher, and remove the class A similar model of AP. The five selected models for ensemble are as follows:

![Speed/accuracy trade-offs for modern convolutional object detectors](https://i.imgur.com/Jun934F.png)


## Insights

- RFCN does not solve the contradiction between positioning and classification. Per ROI subnet should be better, but limit the number of Proposal (mostly negative samples) to reduce redundancy.
- The detection of small objects is still the biggest difficulty. Increasing the resolution and deeper the network does help, but it is not substantial.

## Code
- [TensorFlow](https://github.com/tensorflow/models/tree/master/research/object_detection)


<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<ins class="adsbygoogle"
     style="display:block; text-align:center;"
     data-ad-layout="in-article"
     data-ad-format="fluid"
     data-ad-client="ca-pub-4466575858054752"
     data-ad-slot="8787986126"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>


## Related
- [You Only Look Once: Unified, Real Time Object Detection - Redmon et al. - CVPR 2016](https://arxivnote.ddlee.cn/You-Only-Look-Once-Unified-Real-Time-Object-Detection-Redmon-CVPR-2016.html)
- [An analysis of scale invariance in object detection - SNIP - Singh - CVPR 2018](https://arxivnote.ddlee.cn/An-analysis-of-scale-invariance-in-object-detection-SNIP-Singh-CVPR-2018.html)
- [Faster R-CNN: Towards Real Time Object Detection with Region Proposal - Ren - NIPS 2015](https://arxivnote.ddlee.cn/Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Ren-NIPS-2015.html)
- [(FPN)Feature Pyramid Networks for Object Detection - Lin - CVPR 2017](https://arxivnote.ddlee.cn/Feature-Pyramid-Networks-for-Object-Detection-Lin-CVPR-2017.html)
- [(RetinaNet)Focal loss for dense object detection - Lin - ICCV 2017](https://arxivnote.ddlee.cn/RetinaNet-Focal-loss-for-dense-object-detection-Lin-ICCV-2017.html)
- [YOLO9000: Better, Faster, Stronger - Redmon et al. - 2016](https://arxivnote.ddlee.cn/YOLO9000-Better-Faster-Stronger-Redmon-2016.html)