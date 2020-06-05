---
title: "CVPR 2020: Object Detection(2)"
tag:
- Object Detection
---


## Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector

- Author: Qi Fan, Wei Zhuo, Yu-Wing Tai
- Arxiv: [1908.01998](https://arxiv.org/abs/1908.01998.pdf)
- [GitHub](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset)


### **Problem**
Few-shot object detection: aims to detect objects of unseen class with a few training examples.


### **Insight**
 Central to our method is the Attention-RPN and the multi-relation module which fully exploit the similarity between the few shot training examples and the test set to detect novel objects while suppressing the false detection in background.
 
A new few-shot detection dataset.


### **Technical overview**
![](https://i.imgur.com/f9pD3mW.jpg)

Our network architecture uses ResNet-50 as backbone. The support image (in green) and query image (in blue) are fed into the weight-shared backbone. The RPN use attention feature generated by the depth-wise cross correlation between compact 1 × 1 × C support feature and H ×W × C query feature. The class score generated by patch-relation head (the top head), global-relation head (the middle head) and local-correlation head (the bottom head) is added together as the final matching score, and the bounding box prediction are generated by the patch-relation head. 

![](https://i.imgur.com/4TPiYvs.jpg)
Attention RPN. The support feature is average pooled to a 1× 1×C vector, and then caculate depth-wise cross correlation with the query feature whose output is used as attention feature and is fed into RPN to generate proposals.

**The Dataset**
![](https://i.imgur.com/sPcut4d.jpg)



### **Proof**

![](https://i.imgur.com/uvDGbwe.jpg)



## Dont Even Look Once: Synthesizing Features for Zero-Shot Detection

- Author: Pengkai Zhu, Hanxiao Wang, Venkatesh Saligrama
- Arxiv: [1911.07933](https://arxiv.org/abs/1911.07933.pdf)


### **Problem**
Zero-Shot Detection


### **Assumption in prior work**
At a fundamental level, while vanilla detectors are capable of proposing bounding boxes, which include unseen objects, they are often incapable of assigning high-confidence to unseen objects, due to the inherent precision/recall tradeoffs that requires rejecting background objects. 


### **Technical overview**
Don't Even Look Once (DELO): synthesizes visual features for unseen objects and augments existing training algorithms to incorporate unseen object detection. 
![](https://i.imgur.com/OTHXcKl.jpg)


(a) An illustration of seen/unseen classes and the semantic description; (b) A vanilla detector trained using seen objects only tends to relegate the confidence score of unseen objects; (c) The proposed approach. We first train a visual feature generator by taking a pool of visual features of foreground/background objects and their semantics with a balanced ratio. We then use it to synthesize visual features for unseen objects; Finally we add the synthesized visual features back to the pool and re-train the confidence predictor module of the vanilla detector. The re-trained confidence predictor can be plugged back into the detector and detect unseen objects.


### **Proof**

![](https://i.imgur.com/idpFQwl.jpg)

 Zero-shot detection evaluation results on various datasets and seen/unseen splits. TU = Test-Unseen, TS = Test-Seen, TM = Test-Mix represents different data configurations. Overall average precision (AP) in % is reported. The highest AP for every setting is in bold




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

## Multiple Anchor Learning for Visual Object Detection

- Author: Wei Ke, Tianliang Zhang, Zeyi Huang, Qixiang Ye, Jianzhuang Liu, Dong Huang
- Arxiv: [1912.02252](https://arxiv.org/abs/1912.02252.pdf)


### **Problem**
Classification and localization are two pillars of visual object detectors


### **Assumption in prior work**
In CNN-based detectors, these two modules are usually optimized under a fixed set of candidate (or anchor) bounding boxes. This configuration significantly limits the possibility to jointly optimize classification and localization. 

### **Technical overview**
Our approach, referred to as Multiple Anchor Learning (MAL), constructs anchor bags and selects the most representative anchors from each bag. Such an iterative selection process is potentially NP-hard to optimize. To address this issue, we solve MAL by repetitively depressing the confidence of selected anchors by perturbing their corresponding features. In an adversarial selection-depression manner, MAL not only pursues optimal solutions but also fully leverages multiple anchors/features to learn a detection model. 
![](https://i.imgur.com/ejwZdUd.jpg)

In the feature pyramid network, an anchor bag Ai is constructed for each object bi. Together with the network parameter learning, i.e., back-propagation, MAL evaluates the joint classification and localization confidence of each anchor inAi. Such confidence is used for anchor selection and indicates the importance of anchors during network parameter evolution.
![](https://i.imgur.com/CEOYtXf.jpg)

MAL implementation. During training, it includes the additional anchor selection and anchor depression modules added to RetinaNet. During test, it uses exactly the same architecture as RetinaNet. “U” and “V ” respectively denote convolutional feature maps before and after depression. “M” and “M ′” respectively denote an activation map before and after depression.

### **Proof**
![](https://i.imgur.com/aMmKnkM.jpg)

![](https://i.imgur.com/piUT06j.jpg)



## Rethinking Classification and Localization in R-CNN

- Author: Yue Wu, Yinpeng Chen, Lu Yuan, Zicheng Liu, Lijuan Wang, Hongzhi Li, Yun Fu 
- Arxiv: [1904.06493](https://arxiv.org/abs/1904.06493.pdf)


### **Problem**
Classification and Localization misalignment


### **Assumption in prior work**
Modern R-CNN based detectors share the RoI feature extractor head for both classification and localization tasks, based upon the correlation between the two tasks.


### **Insight**
In contrast, we found that different head structures (i.e. fully connected head and convolution head) have opposite preferences towards these two tasks. Specifically, the fully connected head is more suitable for the classification task, while the convolution head is more suitable for the localization task.


### **Technical overview**
We propose a double-head method to separate these two tasks into different heads (i.e. a fully connected head for classification and a convolution head for box regression).

![](https://i.imgur.com/mfMVpmt.jpg)



### **Proof**
![](https://i.imgur.com/BUPRGF7.jpg)




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

## Revisiting the Sibling Head in Object Detector 
- Author: Guanglu Song, Y. W. Liu, Xiao-gang Wang
- Arxiv: [2003.07540](https://arxiv.org/abs/2003.07540.pdf)
- [GitHub](https://github.com/Sense-X/TSD)


### **Problem**
Classification and Localization misalignment

### **Assumption in prior work**
This paper provides the observation that the spatial misalignment between the two object functions in the sibling head can considerably hurt the training process,


### **Insight**
 Considering the classification and regression, TSD decouples them from the spatial dimension by generating two disentangled proposals for them, which are estimated by the shared proposal. 


### **Technical overview**
![](https://i.imgur.com/kPbJpwK.jpg)

Illustration of the proposed TSD cooperated with Faster RCNN [30]. Input images are first fed into the FPN backbone and then, region proposal P is generated by RPN. TSD adopts the RoI feature of P as input and estimates the derived proposals P̂c and P̂r for classification and localization. Finally, two parallel branches are used to predict specific category and regress precise box, respectively


### **Proof**

![](https://i.imgur.com/3ZY19Lw.jpg)



## DR Loss: Improving Object Detection by Distributional Ranking

- Author: Qi Qian, Lei Chen, Hao Li, Rong Jin
- Arxiv: [1907.10156](https://arxiv.org/abs/1907.10156.pdf)
- [GitHub](https://github.com/idstcv/DR_loss)


### **Problem**: 
Imbalance issue in one-stage detector


### **Assumption in prior work**
One-stage detector is efficient but can suffer from the imbalance issue with respect to two aspects: the imbalance between classes and that in the distribution of background, where only a few candidates are hard to be identified.


### **Technical overview**
First, we convert the classification problem to a ranking problem to alleviate the class-imbalance problem. Then, we propose to rank the distribution of foreground candidates above that of background ones in the constrained worst-case scenario. This strategy not only handles the imbalance in background candidates but also improves the efficiency for the ranking algorithm. 

![](https://i.imgur.com/RZSCfXj.jpg)
First, we re-weight examples to derive the constrained distributions for foreground and background from the original distributions, respectively. Then, we learn to rank the expectation of the derived distribution of foreground above that of background by a large margin.

### **Proof**
![](https://i.imgur.com/9zkt9bn.jpg)



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