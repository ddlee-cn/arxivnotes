---
title: "SNIPER: efficient multi-scale training - Singh - NIPS 2018 - MXNet Code"
tag:
- Object Detection
redirect_from: /SNIPER-efficient-multi-scale-training-Singh-NIPS-2018.html
---



## Info
- Title: **SNIPER: efficient multi-scale training**
- Task: **Object Detection**
- Author: B. Singh, M. Najibi, and L. S. Davis
- Date: May 2018
- Arxiv: [1805.09300](https://arxiv.org/abs/1805.09300)
- Published: NIPS 2018

## Highlights & Drawbacks
- Efficient version of [SNIP](https://arxivnote.ddlee.cn/An-analysis-of-scale-invariance-in-object-detection-SNIP-Singh-CVPR-2018.html) training strategy for object detection
- Select ROIs with proper size only inside a batch

<!-- more -->

## Motivation & Design
![SNIPER: efficient multi-scale training](https://i.imgur.com/fcqlVp8.gif)


Following [SNIP](https://ddleenote.blogspot.com/2019/05/an-analysis-of-scale-invariance-in.html), the authors put crops of an image which contain objects to be detected(called chips) into training instead of the entire image. This design also makes large-batch training possible, which accelerates the training process. This training method utilizes the context of the object, which can save unnecessary calculations for simple background(such as the sky) so that the utilization rate of training data is improved. 

![SNIPER: efficient multi-scale training](https://i.imgur.com/WUTZeG1.png)

The core design of SNIPER is the selection strategy for ROIs from a chip(a crop of entire image). The authors use several hyper-params to filter boxes with proper size in a batch, hopping that the detector network only learns features beyond object size.

Due to its memory efficient design, SNIPER can benefit from Batch Normalization during training and it makes larger batch-sizes possible for instance-level recognition tasks on a single GPU. Hence, there is no need to synchronize batch-normalization statistics across GPUs.

## Performance & Ablation Study
An improvement of the accuracy of small-size objects was reported according to the author’s experiments. 

![SNIPER: efficient multi-scale training](https://i.imgur.com/vQ3Qan1.png)


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

## Code
[MXNet](https://github.com/mahyarnajibi/SNIPER)



## Related
- [You Only Look Once: Unified, Real Time Object Detection - Redmon et al. - CVPR 2016](https://arxivnote.ddlee.cn/You-Only-Look-Once-Unified-Real-Time-Object-Detection-Redmon-CVPR-2016.html)
- [An analysis of scale invariance in object detection - SNIP - Singh - CVPR 2018](https://arxivnote.ddlee.cn/An-analysis-of-scale-invariance-in-object-detection-SNIP-Singh-CVPR-2018.html)
- [Faster R-CNN: Towards Real Time Object Detection with Region Proposal - Ren - NIPS 2015](https://arxivnote.ddlee.cn/Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Ren-NIPS-2015.html)
- [(FPN)Feature Pyramid Networks for Object Detection - Lin - CVPR 2017](https://arxivnote.ddlee.cn/Feature-Pyramid-Networks-for-Object-Detection-Lin-CVPR-2017.html)
- [(RetinaNet)Focal loss for dense object detection - Lin - ICCV 2017](https://arxivnote.ddlee.cn/RetinaNet-Focal-loss-for-dense-object-detection-Lin-ICCV-2017.html)
- [YOLO9000: Better, Faster, Stronger - Redmon et al. - 2016](https://arxivnote.ddlee.cn/YOLO9000-Better-Faster-Stronger-Redmon-2016.html)