---
title: "PolarMask: Single Shot Instance Segmentation with Polar Representation"
tag:
- Instance Segmentation
---

## Info

- Title: PolarMask: Single Shot Instance Segmentation with Polar Representation
- Task: Instance Segmentation
- Author: Enze Xie, Peize Sun , Xiaoge Song, Wenhai Wang, Xuebo Liu, Ding Liang, Chunhua Shen, Ping Luo
- Date: Sep. 2019
- Arxiv: [1909.13226](https://arxiv.org/abs/1909.13226)
- Published: ICCV 2019

## Highlights

- **Simple:** Anchor-free, single-stage and can be easily embeded to many detectors such as FCOS.
- **Unify:** Our PolarMask first make object detection problem and instance segmentation problem into a unified dense regression problem.
- **Inference Fast:** Our PolarMask-R50 can achieve 29.1AP(800) / 23.9FPS, 27.6AP(600) / 34.1FPS, 22.9AP(400) / 46.7FPS on 1 V100 GPU.

## Abstract

In this paper, we introduce an anchor-box free and single shot instance segmentation method, which is conceptually simple, fully convolutional and can be used as a mask prediction module for instance segmentation, by easily embedding it into most off-the-shelf detection methods. Our method, termed PolarMask, formulates the instance segmentation problem as instance center classification and dense distance regression in a polar coordinate. Moreover, we propose two effective approaches to deal with sampling high-quality center examples and optimization for dense distance regression, respectively, which can significantly improve the performance and simplify the training process. Without any bells and whistles, PolarMask achieves 32.9% in mask mAP with single-model and single-scale training/testing on challenging COCO dataset. For the first time, we demonstrate a much simpler and flexible instance segmentation framework achieving competitive accuracy. We hope that the proposed PolarMask framework can serve as a fundamental and strong baseline for single shot instance segmentation tasks. 

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



## Motivation & Design

### Polar Representation

![PolarMask ICCV](https://i.imgur.com/wFgjB7Y.png)


Instance segmentation with different mask representations.

(a) is the original image. (b) is the pixel-wise mask representation. (c) and (d) represent a mask by its contour, in the Cartesian and Polar coordinates, respectively.



### The overall pipeline



![](https://github.com/xieenze/PolarMask/raw/master/imgs/pipeline.png)



The left part contains the backbone and feature pyramid to extract features of different levels. The middle part is the two heads for classification and polar mask regression. H, W, C are the height, width, channels of feature maps, respectively, and k is the number of categories (e.g., k = 80 on the COCO dataset), n is the number of rays (e.g., n = 36).

### Mask IoU

Mask IoU in Polar Representation. Mask IoU (interaction area over union area) in the polar coordinate can be calculated by integrating the differential IoU area in terms of differential angles.


![PolarMask ICCV](https://i.imgur.com/ZwKoMvC.png)



## Experiments & Ablation Study

Note the results are unfair.

![PolarMask ICCV](https://github.com/xieenze/PolarMask/raw/master/imgs/performance.png)

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


[PyTorch](https://github.com/xieenze/PolarMask)



## Related

- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/2019/08/17/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)