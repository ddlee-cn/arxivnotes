---
tile: (OHEM)Training Region-based Object Detectors with Online Hard Example Mining - Shrivastava et al. - CVPR 2016
tag:
- Object Detection
redirect_from: /Training-Region-based-Object-Detectors-with-Online-Hard-Example-Mining-Shrivastava-CVPR-2016.html
---

## Info
- Title: **Training Region-based Object Detectors with Online Hard Example Mining**
- Task: **Object Detection**
- Author: A. Shrivastava, A. Gupta, and R. Girshick
- Date: Apr. 2016
- Arxiv: [1604.03540](https://arxiv.org/abs/1604.03540)
- Published: CVPR 2016

## Highlights & Drawbacks
- Learning-based design for balancing examples for ROI in 2-stage detection network
- Plug-in ready trick, easy to be integrated
- Additional Parameters for Training

<!-- more -->

## Motivation & Design

There is a 1:3 strategy in Faster-RCNN network, which samples negative ROIs(backgrounds) to balance the ratio for positive and negative data in a batch. It's empirical and hand-designed(need additional effort when setting hyper-params).

![ (OHEM)Training Region-based Object Detectors with Online Hard Example Mining](https://i.imgur.com/6aFz3zx.png)

The authors designed an additional sub-network to "learn" the sampling process for negative ROIs, forcing the network focus on ones which are similar to objects(the hard ones), such as backgrounds contain part of objects.

The 'hard' examples are defined using probability from detection head, which means that the sample network is exactly the classification network. In practice, the selecting range is set to [0.1, 0.5].

## Performance & Ablation Study

![ (OHEM)Training Region-based Object Detectors with Online Hard Example Mining](https://i.imgur.com/UyenHVl.png)


![ (OHEM)Training Region-based Object Detectors with Online Hard Example Mining](https://i.imgur.com/ETa4rjl.png)

OHEM can improve performance even after adding bells and whistles like Multi-scale training and Iterative bbox regression.

## Code
[caffe](https://github.com/abhi2610/ohem)


## Related
- [You Only Look Once: Unified, Real Time Object Detection - Redmon et al. - CVPR 2016](https://arxivnote.ddlee.cn/You-Only-Look-Once-Unified-Real-Time-Object-Detection-Redmon-CVPR-2016.html)
- [An analysis of scale invariance in object detection - SNIP - Singh - CVPR 2018](https://arxivnote.ddlee.cn/An-analysis-of-scale-invariance-in-object-detection-SNIP-Singh-CVPR-2018.html)
- [Faster R-CNN: Towards Real Time Object Detection with Region Proposal - Ren - NIPS 2015](https://arxivnote.ddlee.cn/Faster-R-CNN-Towards-Real-Time-Object-Detection-with-Region-Proposal-Ren-NIPS-2015.html)
- [(FPN)Feature Pyramid Networks for Object Detection - Lin - CVPR 2017](https://arxivnote.ddlee.cn/Feature-Pyramid-Networks-for-Object-Detection-Lin-CVPR-2017.html)
- [(RetinaNet)Focal loss for dense object detection - Lin - ICCV 2017](https://arxivnote.ddlee.cn/RetinaNet-Focal-loss-for-dense-object-detection-Lin-ICCV-2017.html)
- [YOLO9000: Better, Faster, Stronger - Redmon et al. - 2016](https://arxivnote.ddlee.cn/YOLO9000-Better-Faster-Stronger-Redmon-2016.html)