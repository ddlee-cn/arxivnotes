---
title: "CVPR 2020: Object Detection(1)"
tag:
- Object Detection
---

## Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection(Oral)

- Author: Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, Stan Z. Li 
- Arxiv: [1912.02424](https://arxiv.org/abs/1912.02424.pdf)
- [GitHub](https://github.com/sfzhang15/ATSS)

### **Problem**
Anchor-based VS. Anchor-Free detectors, what's the true difference

![](https://i.imgur.com/tZqy0XH.png)

### **Insight**
We first point out that the essential difference between anchor-based and anchor-free detection is actually how to define positive and negative training samples, which leads to the performance gap between them. If they adopt the same definition of positive and negative samples during training, there is no obvious difference in the final performance, no matter regressing from a box or a point.

![](https://i.imgur.com/wre7GKS.jpg)

Definition of positives ( 1 ) and negatives ( 0 ). Blue box, red box and red point are ground-truth, anchor box and anchor point. (a) RetinaNet uses IoU to select positives ( 1 ) in spatial and scale dimension simultaneously. (b) FCOS first finds candidate positives ( ? ) in spatial dimension, then selects final positives ( 1 ) in scale dimension.

### **Technical overview**
We propose an Adaptive Training Sample Selection (ATSS) to automatically select positive and negative samples according to statistical characteristics of object.

- Selecting candidates based on the center distance between anchor box and object.
- Using the sum of mean and standard deviation as the IoU threshold.
- Limiting the positive samples’ center to object.Maintaining fairness between different objects.
- Keeping almost hyperparameter-free.


![](https://i.imgur.com/aC1hLvK.png)

### **Proof**

![](https://i.imgur.com/7nMgtKw.jpg)


### **Impact**

Data strategy rule them all. The network structures are barely bells and withles.



## Large-Scale Object Detection in the Wild from Imbalanced Multi-Labels

- Author: Junran Peng, Xingyuan Bu, Ming Sun, Zhaoxiang Zhang, Tieniu Tan, Junjie Yan 
- Arxiv: [2005.08455](https://arxiv.org/abs/2005.08455.pdf)


### **Problem**: 
Imbalanced multi-label in Open Images dataset.


### **Assumption in prior work**
Open Images dataset(from Google) suffers from label-related problems that objects may explicitly or implicitly have multiple labels and the label distribution is extremely imbalanced.


### **Insight**
We design a concurrent softmax to handle the multi-label problems in object detection and propose a softsampling methods with hybrid training scheduler to deal with the label imbalance.


### **Technical overview**
![](https://i.imgur.com/kBB0IyE.jpg)

### **Proof**
![](https://i.imgur.com/avhUuLu.jpg)

![](https://i.imgur.com/vtdEbFw.jpg)



### **Impact**
Some dataset-level strategies on multi-label softmax and sample balance.


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

## NETNet: Neighbor Erasing and Transferring Network for Better Single Shot Object Detection

- Author: Yazhao Li, Yanwei Pang, Jianbing Shen, Jiale Cao, Ling Shao 
- Arxiv: [2001.06690](https://arxiv.org/abs/2001.06690.pdf)


### **Problem**
Scale variance in single-shot detector


### **Assumption in prior work**
 To solve the complex scale variations, single-shot detectors make scale-aware predictions based on multiple pyramid layers. However, the features in the pyramid are not scale-aware enough, which limits the detection performance. Two common problems in single-shot detectors caused by object scale variations can be observed: (1) small objects are easily missed; (2) the salient part of a large object is sometimes detected as an object


### **Insight**
With this observation, we propose a new Neighbor Erasing and Transferring (NET) mechanism to reconfigure the pyramid features and explore scale-aware features. In NET, a Neighbor Erasing Module (NEM) is designed to erase the salient features of large objects and emphasize the features of small objects in shallow layers. A Neighbor Transferring Module (NTM) is introduced to transfer the erased features and highlight large objects in deep layers.


### **Technical overview**

![](https://i.imgur.com/zSQs8sg.jpg)

![](https://i.imgur.com/Z7MjeUc.jpg)

SSD-based

### **Proof**

![](https://i.imgur.com/157WIWf.jpg)


![](https://i.imgur.com/OE4kkZh.jpg)


## Scale-Equalizing Pyramid Convolution for Object Detection

- Author: Xinjiang Wang, Shi-Long Zhang, Zhuoran Yu, Litong Feng, W. Q. Zhang
- Arxiv: [2005.03101](https://arxiv.org/abs/2005.03101.pdf)


### **Problem**: 
across-scale feature fusion in feature pyramid


### **Assumption in prior work**
Early computer vision methods extracted scale-invariant features by locating the feature extrema in both spatial and scale dimension.

Furthermore, we also show that the naive pyramid convolution, together with the design of RetinaNet head, actually best applies for extracting features from a Gaussian pyramid, whose properties can hardly be satisfied by a feature pyramid.


### **Insight**
Inspired by this, a convolution across the pyramid level is proposed in this study, which is termed pyramid convolution and is a modified 3-D convolution. Stacked pyramid convolutions directly extract 3-D (scale and spatial) features and outperforms other meticulously designed feature fusion modules. Based on the viewpoint of 3-D convolution, an integrated batch normalization that collects statistics from the whole feature pyramid is naturally inserted after the pyramid convolution. 

In order to alleviate this discrepancy, we build a scale-equalizing pyramid convolution (SEPC) that aligns the shared pyramid convolution kernel only at high-level feature maps. 

![](https://i.imgur.com/ZOzTKqw.jpg)



### **Technical overview**
![](https://i.imgur.com/vfvj8XH.jpg)
(a) Head design of the original RetinaNet; (b) Head design with PConv.


### **Proof**
![](https://i.imgur.com/lSGodXz.jpg)



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

## AugFPN: Improving Multi-scale Feature Learning for Object Detection

- Author: Chaoxu Guo, Bin Fan, Qian Zhang, Shiming Xiang, Chunhong Pan
- Arxiv: [1912.05384](https://arxiv.org/abs/1912.05384.pdf)
- [GitHub](https://github.com/Gus-Guo/AugFPN)


### **Problem**: 
fusion of multi-scale features in FPN


### **Assumption in prior work**
![](https://i.imgur.com/5o2XnK5.jpg)

Three design defects in feature pyramid network: 1) semantic gap between features at different levels before feature summation, 2) information loss of the feature at the highest pyramid level, 3) heuristic RoI assignment



### **Technical overview**
AugFPN consists of three components: Consistent Supervision, Residual Feature Augmentation, and Soft RoI Selection. AugFPN narrows the semantic gaps between features of different scales before feature fusion through Consistent Supervision. In feature fusion, ratio-invariant context information is extracted by Residual Feature Augmentation to reduce the information loss of feature map at the highest pyramid level. Finally, Soft RoI Selection is employed to learn a better RoI feature adaptively after feature fusion. 

![](https://i.imgur.com/aU3hHab.jpg)


### **Proof**

![](https://i.imgur.com/AfkVdaU.jpg)

![](https://i.imgur.com/ChhIpGZ.jpg)


## Prime Sample Attention in Object Detection

- Author: Yuhang Cao, Kai Chen, Chen Change Loy, Dahua Lin
- Arxiv: [1904.04821](https://arxiv.org/abs/1904.04821.pdf)


### **Problem**: 
performance imbalance among samples


### **Assumption in prior work**
It is a common paradigm in object detection frameworks to treat all samples equally and target at maximizing the performance on average. In this work, we revisit this paradigm through a careful study on how different samples contribute to the overall performance measured in terms of mAP. Our study suggests that the samples in each mini-batch are neither independent nor equally important, and therefore a better classifier on average does not necessarily mean higher mAP.


### **Insight**
We propose the notion of Prime Samples, those that play a key role in driving the detection performance. We further develop a simple yet effective sampling and learning strategy called PrIme Sample Attention (PISA) that directs the focus of the training process towards such samples


### **Technical overview**
![](https://i.imgur.com/Yw8cFP2.jpg)
Left shows both a prime sample (in red color) and a hard sample (in blue color) for an object against the ground-truth. The prime sample has a high IoU with the ground-truth and is located more precisely around the object. Right shows the RoC curves obtained with different sampling strategies, which suggests that attention to prime samples instead of hard samples is a more effective strategy to boost the performance of a detector.

![](https://i.imgur.com/cl10d19.jpg)


### **Proof**
![](https://i.imgur.com/Mn0sFmf.jpg)


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
- [Anchor-Free Object Detection(Part 1): CornerNet, CornerNet-Lite, ExtremeNet, CenterNet](https://arxivnote.ddlee.cn/2019/10/31/Anchor-Free-Object-Detection-CornetNet-CornerNet-Lite-ExtremeNet-CenterNet.html)

- [Anchor-Free Object Detection(Part 2): FSAF, FoveaBox, FCOS, RepPoints](https://arxivnote.ddlee.cn/2019/10/31/Anchor-Free-Object-Detection-FSAF-FoveaBox-FCOS-RepPoints.html)


- [Object Detection Must Reads(1): Fast RCNN, Faster RCNN, R-FCN and FPN](https://arxivnote.ddlee.cn/object-detectin-fast-rcnn-faster-rcnn-rfcn.html)

- [Object Detection Must Reads(2): YOLO, YOLO9000, and RetinaNet](https://arxivnote.ddlee.cn/object-detection-must-reads-yolo-retinanet.html)

- [Object Detection Must Reads(3): SNIP, SNIPER, OHEM, and DSOD](https://arxivnote.ddlee.cn/2019/08/13/Object-Detection-SNIPER-DSOD-OHEM.html)

- [RoIPooling in Object Detection: PyTorch Implementation(with CUDA)](https://cvnote.ddlee.cn/2019/09/16/RoIPooling-PyTorch.html)

- [Bounding Box(BBOX) IOU Calculation and Transformation in PyTorch](https://cvnote.ddlee.cn/2019/09/09/Bounding-Box-Transform.html)

- [Object Detection: Anchor Generator in PyTorch](https://cvnote.ddlee.cn/2019/09/07/PyTorch-Anchor-Generator.html)

- [Assign Ground Truth to Anchors in Object Detection with Python](https://cvnote.ddlee.cn/2019/09/21/Assign-Ground-Truth-Object-Detection.html)

- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)

