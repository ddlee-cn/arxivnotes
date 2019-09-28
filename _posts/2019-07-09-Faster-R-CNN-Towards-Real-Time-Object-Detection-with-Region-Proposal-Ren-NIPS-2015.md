---
title: "Faster R-CNN: Towards Real Time Object Detection with Region Proposal - Ren - NIPS 2015"
tag:
- Object Detection
---

## Info
- Title: **Faster R-CNN: Towards Real Time Object Detection with Region Proposal**
- Task: **Object Detection**
- Author: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- Date: June 2015
- Arxiv: [1506.01497](https://arxiv.org/abs/1506.01497)
- Published: NIPS 2015

## Highlights

Faster R-CNN is the mainstream method of 2-stage method. The proposed RPN network replaces the Selective Search algorithm so that the detection task can be completed end-to-end by the neural network. Roughly speaking, Faster R-CNN = RPN + Fast R-CNN, the nature of the convolution calculation shared with RCNN makes the calculations introduced by RPN very small, allowing Faster R-CNN to run at 5fps on a single GPU. Reach SOTA in terms of accuracy.


<!-- more -->


## Motivation & Design
### Regional Proposal Networks

![Faster R-CNN: Towards Real Time Object Detection with Region Proposal](https://i.imgur.com/Fjlw3aF.png)


The RPN network models the Proposal task as a two-category problem.

The first step is to generate an anchor box of different size and aspect ratio on a sliding window, determine the threshold of the IOU, and calibrate the positive and negative of the anchor box according to Ground Truth. Thus, the sample that is passed into the RPN network is the anchor box and whether there is an object in each anchor box. The RPN network maps each sample to a probability value and four coordinate values. The probability value reflects the probability that the anchor box has an object, and the four coordinate values ​​are used to regress the position of the defined object. Finally, the two classifications and the coordinates of the Loss are unified to be the target training of the RPN network.

The RPN network has a large number of super-parameters, the size and length-to-width ratio of the anchor box, the threshold of IoU, and the ratio of Proposal positive and negative samples on each image.

### Alternate Training

![Faster R-CNN: Towards Real Time Object Detection with Region Proposal](https://i.imgur.com/FlB6GVB.png)


The RPN network is implemented on the feature map, so that the convolution operation of the feature extractor part can be completely shared with the RCNN. During training, the training of the RPN and RCNN can be alternated by alternately fixing the parameters of the RPN and RCNN parts and updating the other part.

## Performance
![Faster R-CNN: Towards Real Time Object Detection with Region Proposal](https://i.imgur.com/6bjIITD.png)
