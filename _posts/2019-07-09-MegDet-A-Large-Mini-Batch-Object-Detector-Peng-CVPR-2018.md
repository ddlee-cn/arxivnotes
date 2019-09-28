---
title: "MegDet: A Large Mini-Batch Object Detector - C. Peng et al - CVPR 2018"
tag:
- Object Detection
---

## Info
- Title: **MegDet: A Large Mini-Batch Object Detector**
- Task: **Object Detection**
- Author: C. Peng et al
- Arxiv: [1711.07240](https://arxiv.org/abs/1711.07240)
- Date: Nov. 2017
- Published: CVPR 2018

## Highlights & Drawbacks
This paper presents a model that despises the first place in 2017 MS COCO Detection chanllenge. A large-scale training and detection network is proposed, and multi-card BN is used to ensure the convergence of the network.

<!-- more -->

## Motivation & Design
The article first pointed out that the above methods are mostly updates of the framework, loss, etc., and all use very small batch (2 pictures) training, which has the following shortcomings:

- training slow
- fails to provide accurate statistics for BN

This involves a question, whether the source data of the detection task should be a picture or a label box. In Fast R-CNN, RBG mentioned that the labeling box of each batch sample such as SPPNet comes from different pictures, and the convolution operation cannot be shared between them (the convolution operation is in pictures). In order to share this part of the calculation, Fast R-CNN adopted the strategy of “select the picture first, then select the label box” to determine each batch. The article mentioned that this kind of operation will introduce relevance, but it has little effect in practice. After the Faster R-CNN, each picture generated about 300 Proposal via RPN, and the introduction of RCNN has become a common practice.

Personally think that the data of the inspection task should be based on pictures. Objects produce semantics in the background of the image, and although each image has multiple Proposals (approximating the batch size in the classification task), they share the same semantics (scene), and a single semantic is difficult to be in the same Batch provides diversity for online learning.

### Dilemma
Increasing mini-batch size requires large learning rate, which may cause discovergence.

### solution
- new explanation of linear scaling rule, introduce "warmup" trick to learning rate schedule
- Cross GPU Batch Normalization (CGBN)

### Variance Equivalence explanation for Linear Scaling Rule
The linear scaling rule comes from changing the batch size and scaling the learning rate so that the changed weight update is similar to the previous small batch size, multi-step weight update. In this paper, the linear scaling rule is reinterpreted by keeping the variance of the loss gradient unchanged, and it is pointed out that this assumption only requires the loss gradient to be i.i.d, which is similar to the loss gradient between different batch sizes assumed to be kept by weight update.

### WarmUp Strategy
In the early stage of training, the weight jitter is obvious. The warmup mechanism is introduced to use the smaller learning rate, and then gradually increase to the learning rate required by the Linear scaling rule.

### Cross-GPU Batch Normalization
BN is one of the key technologies to enable deep network training and convergence. However, in the detection task, the fine-tuning phase often fixes the BN part parameters of the SOTA classification network and does not update.

Larger resolution images are often needed for detection, while GPU memory limits the number of pictures on a single card. Increasing the batch size means that BN is going to be done on a multi-card (Cross-GPU).

The BN operation needs to calculate the mean and variance for each batch to standardize. For multi-card, the specific method is to calculate the average value by a single card, and aggregate (similar to Reduce in Map-Reduce) to calculate the mean, and then send the average to each. Cards, calculate the difference, then aggregate them, calculate the variance of the batch, and finally send the variance to each card, and normalize it with the average value of the previous delivery.

The process is as follows:

![MegDet: A Large Mini-Batch Object Detector](https://i.imgur.com/XW5B524.png)

## Performance & Ablation Study

The architecture on the COCO dataset uses the pre-trained ResNet-50 as the underlying network, and the FPN is used to provide the feature map.

The results show that the larger batch size (64, 128) does not converge when BN is not used. After using BN, increasing the Batch size can converge but only brings a small increase in accuracy, and the size of the BN is not as large as possible. In the experiment, 32 is the best choice. The main results are as follows:

![MegDet: A Large Mini-Batch Object Detector](https://i.imgur.com/nzg4JVQ.png)

Press epoch, the accuracy changes as shown below, small batch (16) in the first few epoch performance is better than the big batch (32).

![MegDet: A Large Mini-Batch Object Detector](https://i.imgur.com/1yMOlT1.png)



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