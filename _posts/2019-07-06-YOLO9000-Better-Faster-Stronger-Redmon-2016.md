---
title: "YOLO9000: Better, Faster, Stronger - Redmon et al. - 2016"
tag:
- Object Detection
---



## Info

- Title: **YOLO9000: Better, Faster, Stronger**
- Task: **Object Detection**
- Author: J. Redmon and A. Farhadi
- Arxiv: [1612.08242](https://arxiv.org/abs/1612.08242)
- Date: Dec. 2016

## Highlights & Drawbacks
A significant improvement for [YOLO](https://ddleenote.blogspot.com/2019/05/you-only-look-once-unified-real-time.html).

<!-- more -->

## Motivation & Design
1. Add BN to the convolutional layer and discard Dropout
2. Higher size input
3. Use Anchor Boxes and replace the fully connected layer with convolution in the head
4. Use the clustering method to get a better a priori for generating Anchor Boxes
5. Refer to the Fast R-CNN method for log/exp transformation of position coordinates to keep the loss of coordinate regression at the appropriate order of magnitude.
6. Passthrough layer: Similar to ResNet's skip-connection, stitching feature maps of different sizes together
7. Multi-scale training
8. More efficient network Darknet-19, a VGG-like network, achieves the same accuracy as the current best on ImageNet with fewer parameters.

After this improvement, YOLOv2 absorbs the advantages of a lot of work, achieving the accuracy and faster inference speed of SSD.

The author also introduces a new joint training method: training the classification task and the detection task at the same time, so that the detection model can be generalized to the target class outside the detection training set.

YOLO9000 uses the joint training of ImageNet and COCO datasets. When combining the labels of the two, the tree-like category prediction map is constructed according to the inheritance relationship of WordNet:

![YOLO9000: Better, Faster, Stronger](https://i.imgur.com/GqLMOS9.jpg)

The probability value of each sub-tag is calculated in a manner similar to the conditional probability. When a certain threshold is exceeded, the class is selected as the output, and only the loss calculation and BP are performed on the class on the path.

YOLO9000 provides us with a training method for generalized detection models. The results of the article show that YOLO9000 has about 20 mAP performances in categories without COCO labeling, and more than 9,000 object types can be detected. Of course, its generalization performance is also constrained by the type of detected annotations, which performs well on classes with class inheritance relationships, and performs poorly on classes that have no semantic association at all.

## Performance & Ablation Study
![YOLO9000: Better, Faster, Stronger](https://i.imgur.com/1G7OXeq.jpg)

## Code
- [Project Site(Contains newest v3)](https://pjreddie.com/darknet/yolo/)
- [Darknet](https://github.com/pjreddie/darknet)