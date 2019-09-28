---
title: An analysis of scale invariance in object detection - SNIP - Singh - CVPR 2018
tag:
- Object Detection
---



## Info
- Title: **An analysis of scale invariance in object detection - SNIP**
- Task: **Object Detection**
- Author: B. Singh and L. S. Davis
- Date: Nov. 2017
- Arxiv: [1711.08189](https://arxiv.org/abs/1711.08189)
- Published: CVPR 2018

## Highlights & Drawbacks
- Training strategy optimization, ready to integrate with other tricks
- Informing experiments for multi-scale training trick

<!-- more -->

## Motivation & Design

![An analysis of scale invariance in object detection](https://i.imgur.com/WN1uG4W.png)

In Object Detection challenges such as COCO, multi-scale training is often used as a trick which is believed to improve the detection accuracy of small-size object significantly. 

![An analysis of scale invariance in object detection](https://i.imgur.com/rn0jAf0.png)


However, the author’s experiments show that resizing images has little impact on the detection accuracy for all sizes on the whole. The author believes that the network handles scale information by simply memorizing. To solve this problem, they designed a training method which filters ground truth with similar size in the same training iteration, instead of using the data of large and small objects at the same time.

![An analysis of scale invariance in object detection](https://i.imgur.com/xqGjsM1.png)


The process of SNIP:
1. Select 3 image resolutions: (480, 800) to train [120, ∞) proposals, (800, 1200) to train [40, 160] proposals, (1400, 2000) to train [0, 80] for proposals

2. For each resolution image, BP only returns the gradient of the proposal within the corresponding scale.

3. This ensures that only one network is used, but the size of each training object is the same, and the size of the object of ImageNet is consistent to solve the problem of domain shift, and it is consistent with the experience of the backbone, and the training and test dimensions are consistent, satisfying " ImageNet pre-trained size, an object size, a network, a receptive field, these four match each other, and the train and test dimensions are the same.

4. A network, but using all the object training, compared to the scale specific detector, SNIP is fully conducive to the data

5. During the test, the same detector is measured once on each of the three resolution images, and only the detected boxes of the corresponding scale are retained for each resolution image, and then merged to execute SoftNMS.

## Performance & Ablation Study

The authors conducted experiments for RFCN and Faster-RCNN and SNIP improves performance for both meta architetures.

![An analysis of scale invariance in object detection](https://i.imgur.com/6gKN6kw.png)



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