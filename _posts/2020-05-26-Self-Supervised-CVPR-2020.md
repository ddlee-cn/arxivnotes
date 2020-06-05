---
title: "CVPR 2020: Self-Supervised Learning"
tag:
- Self-Supervised
---

## Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics(Oral)

- Author: Simon Jenni, Hailin Jin, Paolo Favaro
- Arxiv: [2004.02331](https://arxiv.org/abs/2004.02331.pdf)


### **Problem**
Recognizing the pose of objects from a single image that for learning uses only unlabelled videos and a weak empirical prior on the object poses.


### **Insight**

Prevent appearance leakage in CycleGAN through:
(a) novel bottleneck with a differentiable sketch renderer.
(b) Conditioning the generator on an appearance image.


We introduce a whole new design for the model and for the ‘pose bottleneck’. In particular, we adopt a dual representation of pose as a set of 2D object coordinates, and as a pictorial representation of the 2D coordinates in the form of a skeleton image.


### **Technical overview**
![](https://i.imgur.com/c4CVFZh.jpg)

We learn an encoder Φ that maps an image x to its pose y, represented as a skeleton image. This is done via conditional auto-encoding, learning also a decoder Ψ that reconstruct the input x from its pose y and a second auxiliary video frame x ′ . A bottleneck β ◦ η is used to drop appearance information that may leak in the pose image y. A discriminator D is used to match the distribution of predicted poses to a reference prior distribution, represented by unpaired pose samples ¯y.


### **Proof**
- Datasets: Huamn3.6M, PennAction, MPI_INF_3DHP
- Baselines: Self-supervised + full supervised
- Metrics: mean error of keypointss


### **Impact**
Pose detector + Image translation

Disentangling style and geometry

![](https://i.imgur.com/UPTclCH.jpg)

## Steering Self-Supervised Feature Learning Beyond Local Pixel Statistics(Oral)

- Author: Simon Jenni, Hailin Jin, Paolo Favaro
- Arxiv: [2004.02331](https://arxiv.org/abs/2004.02331.pdf)
- [GitHub](https://sjenni.github.io/LCI)


### **Problem**



### **Assumption in prior work**
Classifiers: It might yield features that describe mostly local statistics, and thus have limited generalization capabilities.

Training supervised models to focus on the global statistics (which they refer to as shape) can improve the generalization and the robustness of the learned image representation.

### **Insight**
We argue that the generalization capability of learned features depends on what image neighborhood size is sufficient to discriminate different image transformations: The larger the required neighborhood size and the more global the image statistics that the feature can describe.

We introduce a novel image transformation that we call limited context inpainting (LCI). This transformation inpaints an image patch conditioned only on a small rectangular pixel boundary (the limited context). Because of the limited boundary information, the inpainter can learn to match local pixel statistics, but is unlikely to match the global statistics of the image.

The training task in our method is to discriminate global image statistics. To this end, we transform images in such a way that local statistics are largely unchanged, while global statistics are clearly altered. By doing so, we make sure that the discrimination of such transformations is not possible by working on just local patches, but instead it requires using the whole image.

### **Technical overview**
LCI selects a random patch from a natural image, substitutes the center with noise (thus, it preserves a small outer boundary of pixels), and trains a network to inpaint a realistic center through adversarial training. While LCI can inpaint a realistic center of the patch so that it seamlessly blends with the preserved boundaries, it is unlikely to provide a meaningful match with the rest of the original image. Hence, this mismatch can only be detected by learning global statistics of the image.
![](https://i.imgur.com/o2o48q2.jpg)

Training of the Limited Context Inpainting (LCI) network. A random patch is extracted from a training image x and all but a thin border of pixels is replaced by random noise. The inpainter network F fills the patch with realistic textures conditioned on the remaining border pixels. The resulting patch is replaced back into the original image, thus generating an image with natural local statistics, but unnatural global statistics.


### **Proof**

- Datasets: ImageNet Classification, Pascal VOC Det and Seg
- Metrics: mAP, mIOU, Nearest neighbor retrieval
- Baselines: self-supervised methods

Transfer Learning results:

![](https://i.imgur.com/AJjO8GF.jpg)


Experiments shows that this is better(global) that warping and rotation.
![](https://i.imgur.com/s9bh2m2.jpg)

### **Impact**

Nice pseudo supervision.

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



## Self-Supervised Scene De-occlusion

- Author: Xiaohang Zhan, Xingang Pan, Bo Dai, Ziwei Liu, Dahua Lin, Chen Change Loy
- Arxiv: [123.123](https://arxiv.org/pdf/2004.02788.pdf)


### **Problem**
Scene de-occlusion: aims to recover the underlying occlusion ordering and complete the invisible parts of occluded objects. 


### **Insight**
We make the first attempt to address the problem through a novel and unified framework that recovers hidden scene structures without ordering and amodal annotations as supervisions. This is achieved via Partial Completion Network (PCNet)-mask (M) and -content (C), that learn to recover fractions of object masks and contents, respectively, in a self-supervised manner. Based on PCNet-M and PCNet-C, we devise a novel inference scheme to accomplish scene de-occlusion, via progressive ordering recovery, amodal completion and content completion. 


### **Technical overview**
![](https://i.imgur.com/0qrKcLr.jpg)

The training procedure of the PCNet-M and the PCNet-C. Given an instance A as the input, we randomly sample another instance B from the whole dataset and position it randomly. Note that we only have modal masks of both A and B. (a) PCNet-M is trained by switching two cases. Case 1 (A erased by B) follows the partial completion mechanism where PCNet-M is encouraged to partially complete A. Case 2 prevents PCNet-M from over completing A. (b) PCNet-C uses A ∩ B to erase A and learn to fill in the RGB content of the erased region. It also takes in A\B as an additional input. The modal mask of A is multiplied with its category id if available.

![](https://i.imgur.com/UPlNXZp.jpg)
(a) Ordering-grounded amodal completion takes the modal mask of the target object (#3) and all its ancestors (#2, #4), as well as the erased image as inputs. With the trained PCNetM, it predicts the amodal mask of object #3. (b) The intersection of the amodal mask and the ancestors indicates the invisible region of object #3. Amodal-constrained content completion (red arrows) adopts the PCNet-C to fill in the content in the invisible region


## 

- Author:
- Arxiv: [123.123](https://arxiv.org/abs/)


### **Problem**
Representation learning of sequential data(video and audio)

### **Insight**
With the supervision of the signals, our model can easily disentangle the representation of an input sequence into static factors and dynamic factors (i.e., time-invariant and time-varying parts).


### **Technical overview**
![](https://i.imgur.com/aO0jlSD.jpg)

The framework of our proposed model in the context of video data. Each frame of a video x1:T is fed into an encoder to produce a sequence of visual features, which is then passed through an LSTM module to obtain the manifold posterior of a dynamic latent variable {q(zt|x≤t)}Tt=1 and the posterior of a static latent variable q(zf |x1:T ). The static and dynamic representations zf and z1:T are sampled from the corresponding posteriors and concatenated to be fed into a decoder to generate reconstructed sequence x̃1:T . Three regularizers are imposed on dynamic and static latent variables to encourage the representation disentanglement.

### **Proof**

- Datasets: Moving MNIST, Sprite, MUG Facial Experssion
- Tasks: Representation Swapping, Video Generation
- Metrics: IS, Intra-Entropy, Inter-Entropy
- Baseline: DSVAE, MoCoGAN


![](https://i.imgur.com/5I8nhRE.jpg)


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


## How Useful is Self-Supervised Pretraining for Visual Tasks?

- Author: Alejandro Newell, Jun Deng
- Arxiv: [2003.14323](https://arxiv.org/abs/2003.14323.pdf)
- [GitHub](https://github.com/princeton-vl/selfstudy)


### **Problem**
Evaluate the effectiveness of self-supervised pretraining for downstream tasks.

### **Insight**
Three possible outcomes:
a) always provides an improvement over the the model trained from scratch even as the amount of labeled data increases, 
b) reaches higher accuracy with fewer labels but plateaus to the same accuracy as the baseline, 
c) converges to baseline performance before accuracy plateaus. 

In our experiments we find option (c) to be the most common outcome.

![](https://i.imgur.com/fpwqXnv.jpg)



### **Technical overview**
Synthetic bechmark: Our synthetic images consist of objects ﬂoating in empty space. For a given image, we can change the number of objects, their orientation, their texture, as well as the lighting conditions of the scene. We render images with Blender [7] using object models from ShapeNet [5]. We choose 10 object classes to use in all versions of the synthetic data (airplane, bench, cabinet, car, chair, lamp, sofa, table, watercraft, motorcycle).

Variations: texture, color, viewpoint, lighting.

Downstream tasks: object classification, object pose estimation, semantic segmentation, depth estimation


### **Proof**
![](https://i.imgur.com/OeOyrzZ.jpg)

Object classification accuracy and utility of pretrained ResNet9 models when finetuning on increasing numbers of labeled samples. As more labeled data is included, the utility (ratio of labels saved) tends toward zero eventually converging with performance when trained from scratch. This occurs before model performance has saturated.


### **Impact**
For the synthetic dataset, representations for specific task can be learned from full supervised learning. Self-supervised pretraining can hardly improve the performance.

But, in real-world data? (Kaiming's rethinking ImageNet pretraining)


## Self-Supervised Learning of Pretext-Invariant Representations

- Author: Ishan Misra, Laurens van der Maaten
- Arxiv: [1912.01991](https://arxiv.org/abs/1912.01991.pdf)
- [GitHub](https://github.com/akwasigroch/Pretext-Invariant-Representations)


### **Problem**
Representation learning via self-supervised training

### **Insight**
 Many pretext tasks lead to representations that are covariant with image transformations. We argue that, instead, semantic representations ought to be invariant under such transformations. 

![](https://i.imgur.com/JW1gpRW.jpg)


### **Technical overview**
![](https://i.imgur.com/zFm9ZS0.jpg)

Overview of PIRL. Pretext-Invariant Representation Learning (PIRL) aims to construct image representations that are invariant to the image transformations t ∈ T . PIRL encourages the representations of the image, I, and its transformed counterpart, It, to be similar. It achieves this by minimizing a contrastive loss. Following [72], PIRL uses a memory bank,M, of negative samples to be used in the contrastive learning. The memory bank contains a moving average of representations, mI ∈M, for all images in the dataset

### **Proof**

- Tasks: ImageNet Classification, Pascal VOC Detection

![](https://i.imgur.com/X5dWfgt.jpg)


![](https://i.imgur.com/5zwSGcg.jpg)



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

- [Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://arxivnote.ddlee.cn/2019/10/11/Temporal-Cycle-Consistency-Learning.html)
- [Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://arxivnote.ddlee.cn/2019/09/29/Learning-Correspondence-Cycle-Consistency-Time.html)
