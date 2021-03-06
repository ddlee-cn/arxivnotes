---
title: "Object Detection Must Reads(Part 2): YOLO, YOLO9000, and RetinaNet"
tag:
- Object Detection
- Review
redirect_from: /object-detection-must-reads-yolo-retinanet.html
---



In [previous article]((https://arxivnote.ddlee.cn/object-detectin-fast-rcnn-faster-rcnn-rfcn.html)), we reviewed 2-stage state-of-art object detectors: Fast RCNN, Faster RCNN, R-FCN, and FPN. We'll introduce 1-stage object detection models in this one.

## (YOLO)You Only Look Once: Unified, Real Time Object Detection - Redmon et al. - CVPR 2016

### Info
- Title: **You Only Look Once: Unified, Real Time Object Detection**
- Task: **Object Detection**
- Author: J. Redmon, S. Divvala, R. Girshick, and A. Farhadi
- Arxiv: https://arxiv.org/abs/1506.02640
- Date: June. 2015
- Published: CVPR 2016

### Highlights & Drawbacks
- Fast.
- Global processing makes background errors relatively small compared to local (regional) based methods such as Fast RCNN.
- Generalization performance is good, YOLO performs well when testing on art works.
- The idea of YOLO meshing is still relatively rough, and the number of boxes generated by each mesh also limits its detection of small objects and similar objects.


<!-- more -->

### Design

![You Only Look Once: Unified, Real Time Object Detection](https://i.imgur.com/ZO5EiVs.png)


![You Only Look Once: Unified, Real Time Object Detection](https://i.imgur.com/dBqrPc5.png)



The loss function is divided into three parts: coordinate error, object error, and class error. In order to balance the effects of category imbalance and large and small objects, weights are added to the loss and the root length is taken.

### Performance & Ablation Study

![You Only Look Once: Unified, Real Time Object Detection](https://i.imgur.com/RJQH4lU.png)




![You Only Look Once: Unified, Real Time Object Detection](https://i.imgur.com/Bx7fLGT.png)

Compared to Fast-RCNN, YOLO's background false detections account for a small proportion of errors, while position errors account for a large proportion (no log coding).

## Code
- [Project Site(Contains newest v3)](https://pjreddie.com/darknet/yolo/)
- [Darknet](https://github.com/pjreddie/darknet)

Check full introduction at [You Only Look Once: Unified, Real Time Object Detection - Redmon et al. - CVPR 2016](https://arxivnote.ddlee.cn/You-Only-Look-Once-Unified-Real-Time-Object-Detection-Redmon-CVPR-2016.html).

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

## YOLO9000: Better, Faster, Stronger - Redmon et al. - 2016

### Info
- Title: **YOLO9000: Better, Faster, Stronger**
- Task: **Object Detection**
- Author: J. Redmon and A. Farhadi
- Arxiv: [1612.08242](https://arxiv.org/abs/1612.08242)
- Date: Dec. 2016

### Highlights & Drawbacks
A significant improvement for [YOLO](https://arxivnote.ddlee.cn/You-Only-Look-Once-Unified-Real-Time-Object-Detection-Redmon-CVPR-2016.html)).

### Design
1. Add BN to the convolutional layer and discard Dropout
2. Higher size input
3. Use Anchor Boxes and replace the fully connected layer with convolution in the head
4. Use the clustering method to get a better a priori for generating Anchor Boxes
5. Refer to the Fast R-CNN method for log/exp transformation of position coordinates to keep the loss of coordinate regression at the appropriate order of magnitude.
6. Passthrough layer: Similar to ResNet's skip-connection, stitching feature maps of different sizes together
7. Multi-scale training
8. More efficient network Darknet-19, a VGG-like network, achieves the same accuracy as the current best on ImageNet with fewer parameters.

After this improvement, YOLOv2 absorbs the advantages of a lot of work, achieving the accuracy and faster inference speed of SSD.

### Performance & Ablation Study
![YOLO9000: Better, Faster, Stronger](https://i.imgur.com/1G7OXeq.jpg)

### Code
- [Project Site(Contains newest v3)](https://pjreddie.com/darknet/yolo/)
- [Darknet](https://github.com/pjreddie/darknet)

Check full introduction at [YOLO9000: Better, Faster, Stronger - Redmon et al. - 2016](https://arxivnote.ddlee.cn/YOLO9000-Better-Faster-Stronger-Redmon-2016.html).

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


## (RetinaNet)Focal loss for dense object detection - Lin  - ICCV 2017

### Info
- Title: **Focal loss for dense object detection**
- Task: **Object Detection**
- Author: T. Lin, P. Goyal, R. B. Girshick, K. He, and P. Dollár
- Date: Aug. 2017
- Arxiv: [1708.02002](https://arxiv.org/abs/1708.02002)
- Published: ICCV 2017(Best Student Paper)

### Highlights & Drawbacks
- Loss function improvement
- For *Dense* samples from single-stage models like SSD

###  Design

In single-stage models, a massive number of training samples are calculated in the loss function at the same time, because of the lack of proposing candidate regions. Based on findings that the loss of a single-stage model is dominated by easy samples(usually backgrounds), Focal Loss introduces a suppression factor on losses from these easy samples, in order to let hard cases play a bigger role in the training process. 


![(RetinaNet)Focal loss for dense object detection](https://i.imgur.com/C6uuJrQ.png)

Utilizing focal loss term, a dense detector called RetinaNet is designed based on ResNet and FPN:

![(RetinaNet)Focal loss for dense object detection](https://i.imgur.com/62SFpNT.png)


### Performance & Ablation Study
The author’s experiments show that a single-stage detector can achieve comparable accuracy like two-stage models thanks to the proposed loss function. However, Focal Loss function brings in two additional hyper-parameters. The authors use a grid search to optimize these two hyper-parameters, which is not inspiring at all since it provides little experience when using the proposed loss function on other datasets or scenarios. Focal Loss optimizes the weight between easy and hard training samples in the loss function from the perspective of sample imbalance.

![(RetinaNet)Focal loss for dense object detection](https://i.imgur.com/IObdCS1.png)


## Code
[Caffe2(FAIR's Detectron)](https://github.com/facebookresearch/Detectron)

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
- [Object Detection Must Reads(1): Fast RCNN, Faster RCNN, R-FCN and FPN](https://arxivnote.ddlee.cn/object-detectin-fast-rcnn-faster-rcnn-rfcn.html)

- [Object Detection Must Reads(3): SNIP, SNIPER, OHEM, and DSOD](https://arxivnote.ddlee.cn/object-detection-must-reads-snip-sniper-ohem-dsod.html)

- [RoIPooling in Object Detection: PyTorch Implementation(with CUDA)](https://cvnote.ddlee.cn/2019/09/16/RoIPooling-PyTorch.html)

- [Bounding Box(BBOX) IOU Calculation and Transformation in PyTorch](https://cvnote.ddlee.cn/2019/09/09/Bounding-Box-Transform.html)

- [Object Detection: Anchor Generator in PyTorch](https://cvnote.ddlee.cn/2019/09/07/PyTorch-Anchor-Generator.html)

- [Assign Ground Truth to Anchors in Object Detection with Python](https://cvnote.ddlee.cn/2019/09/21/Assign-Ground-Truth-Object-Detection.html)

- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)

- [Convolutional Neural Network Must Reads: Xception, ShuffleNet, ResNeXt and DenseNet](https://arxivnote.ddlee.cn/convolutional-neural-network-xception-shufflenet-resnext-densenet.html)

- [Anchor-Free Object Detection(Part 1): CornerNet, CornerNet-Lite, ExtremeNet, CenterNet](https://arxivnote.ddlee.cn/2019/10/31/Anchor-Free-Object-Detection-CornetNet-CornerNet-Lite-ExtremeNet-CenterNet.html)

- [Anchor-Free Object Detection(Part 2): FSAF, FoveaBox, FCOS, RepPoints](https://arxivnote.ddlee.cn/2019/10/31/Anchor-Free-Object-Detection-FSAF-FoveaBox-FCOS-RepPoints.html)

  