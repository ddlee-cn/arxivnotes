---
title: "CVPR 2020: Image-to-Image Translation(2)"
tag:
- Imaget-to-Image Translation
- GAN
---

## (CoCosNet) Cross-domain Correspondence Learning for Exemplar-based Image Translation

- Author: Pan Zhang, Bo Zhang, Dong Chen, Lu Yuan, Fang Wen
- Arxiv: [2004.05571](https://arxiv.org/abs/2004.05571)
- [Project Site](https://panzhang0212.github.io/CoCosNet/)


### **Problem**
exemplar-based image translation


### **Assumption in prior work**
Previous exemplar-based method only use style code globally.
The style code only characterizes the global style of the exemplar, regardless of spatial relevant information. Thus, it causes some local style “wash away” in the ultimate image.

Deep Image Analogy is not cross-domain, may fail to handle a more challenging mapping from mask (or edge, keypoints) to photo since the pretrained network does not recognize such images.

### **Insight**

![](https://i.imgur.com/EOLx82w.jpg)


With the cross-domain correspondence, we present a general solution to exemplar-based image translation, that for the first time, outputs images resembling the fine structures of the exemplar at instance level.

### **Technical overview**

![](https://i.imgur.com/FtdEk3G.jpg)


The network architecture comprises two sub-networks: 1) Cross-domain correspondence Network transforms the inputs from distinct domains to an intermediate feature domain where reliable dense correspondence can be established; 2) Translation network, employs a set of spatially-variant de-normalization blocks [38] to progressively synthesizes the output, using the style details from a warped exemplar which is semantically aligned to the mask (or edge, keypoints map) according to the estimated correspondence.


### **Proof**

- Datasets: ADE20k, CelebA-HQ, Deepfashion
- Baselines: Pix2pixHD, SPADE, MUNIT, SIMS, EGSC-IT
- Metrics: 1)FID, SWD; 2)semantic consistency: high-level feature distance from ImageNet VGG; 3)style relevance: low-level feature distance from ImageNet VGG 4)User Study ranking

![](https://i.imgur.com/IAk984j.jpg)


### **Impact**

Applications: Image editing, Makeup transfer

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

## StarGAN v2: Diverse Image Synthesis for Multiple Domains

- Author: Yunjey Choi, Youngjung Uh, JaeJun Yoo, Jungwoo Ha
- Arxiv: [1912.01865](https://arxiv.org/abs/1912.01865.pdf)
- [GitHub-PyTorch](https://github.com/clovaai/stargan-v2), [GitHub-TensorFlow](https://github.com/taki0112/StarGAN_v2-Tensorflow)


### **Problem**
multiple domain image translation


### **Assumption in prior work**
Existing Image-to-image translation methods have only considered a mapping between two domains, they are not scalable to the increasing number of domains. For example, having K domains, these methods require to train K(K-1) generators to handle translations between each and every domain, limiting their practical usage.

StarGAN still learns a deterministic mapping per each domain, which does not capture the multi-modal nature of the data distribution. 


### **Insight**

![](https://i.imgur.com/BkHYRKw.jpg)

generate diverse images across multiple domains.


### **Technical overview**

![](https://i.imgur.com/I1yp3uT.jpg)

(a) The generator translates an input image into an output image reflecting the domain-specific style code. (b) The mapping network transforms a latent code into style codes for multiple domains, one of which is randomly selected during training. (c) The style encoder extracts the style code of an image, allowing the generator to perform reference-guided image synthesis. (d) The discriminator distinguishes between real and fake images from multiple domains

In particular, we start from StarGAN and replace its domain label with our proposed domain-specific style code that can represent diverse styles of a specific domain. To this end, we introduce two modules, a mapping network and a style encoder. The mapping network learns to transform random Gaussian noise into a style code, while the encoder learns to extract the style code from a given reference image. Considering multiple domains, both modules have multiple output branches, each of which provides style codes for a specific domain. Finally, utilizing these style codes, our generator learns to successfully synthesize diverse images over multiple domains

### **Proof**
- Datasets: CelebA-HQ, AFHQ(new)
- Baselines: MUNIT, DRIT, MSGAN, StarGAN
- Metrics: FID, LPIPS

![](https://i.imgur.com/qL2aX2s.jpg)


### **Impact**

reference-guided synthesis


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


## Panoptic-based Image Synthesis

- Author: Aysegul Dundar, Karan Sapra, Guilin Liu, Andrew Tao, Bryan Catanzaro
- Arxiv: [2004.10289](https://arxiv.org/abs/2004.10289.pdf)


### **Problem**
from panoptic map to image


### **Assumption in prior work**
Previous conditional image synthesis algorithms mostly rely on semantic maps, and often fail in complex environments where multiple instances occlude each other.
This is the result of conventional convolution and upsampling algorithms being independent of class and instance boundaries.

### **Insight**
We are interested in panoptic maps because semantic maps do not provide sufficient information to synthesize “things” (instances) especially in complex environments with multiple of them interacting with each other.

We propose Panoptic aware upsampling that addresses the misalignment between the upsampled low resolution features and high resolution panoptic maps. This ensures that the semantic and instance details are not lost, and that we also maintain higher accuracy alignment between the generated images and the panoptic maps.

![](https://i.imgur.com/J9JZ0dP.jpg)


### **Technical overview**

![](https://i.imgur.com/I1oHWBQ.jpg)

**Panoptic Aware Convolution Layer**
![](https://i.imgur.com/wPZumt0.jpg)

Panoptic aware partial convolution layer takes a panoptic map (colorized for visualization) and based on the center of each sliding window it generates a binary mask, M. The pixels that share the same identity with the center of the window are assigned 1 and the others 0.

**Panoptic Aware Upsampling Layer**

![](https://i.imgur.com/LHaP3qk.jpg)

As shown in Figure (top), first we correct misalignment by replicating a feature vector from a neighboring pixel that belongs to the same panoptic instance. This operation is different from nearest neighbor upsampling which would always replicate the top-left feature. Second, as shown in Figure (bottom), we resolve pixels where new semantic or instance classes have just appeared by encoding new features from semantic maps with Panoptic aware convolution layer.

### **Proof**
- Datasets: COCO-Stuff, Cityscapes
- Baselines: CRN, SIMS, SPADE
- Metrics: detAP(for instance detection), mIoU, pixel Acc, FID



![](https://i.imgur.com/QK1Z4mw.jpg)



### **Impact**


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

## SketchyCOCO: Image Generation from Freehand Scene Sketches

- Author: Chengying Gao, Qi Liu, Qi Xu, Jianzhuang Liu, Li-Jae Wang, Changqing Zou
- Arxiv: [2003.02683](https://arxiv.org/abs/2003.02683.pdf)

### **Problem**

![](https://i.imgur.com/5lIUpcu.jpg)

controllably generating realistic images with many objects and relationships from a freehand scene-level sketch


### **Assumption in prior work**
The author argue that this is a new problem.


### **Insight**
Using freehand sketches as conditional signal is hard.


### **Technical overview**

Two sequential stages, foreground and background generation, based on the characteristics of scene-level sketching. The first stage focuses on foreground generation where the generated image content is supposed to exactly meet the user’s specific requirement. The second stage is responsible for background generation where the generated image content may be loosely aligned with the sketches.

**Dataset: SketchyCOCO**
![](https://i.imgur.com/mTOwMj5.jpg)

Illustration of five-tuple ground truth data of SketchyCOCO, i.e., 
(a) {foreground image, foreground sketch, foreground edge maps} (training: 18,869, test: 1,329), 
(b) {background image, background sketch} (training: 11,265, test: 2,816), 
(c) {scene image, foreground image & background sketch} (training: 11,265, test: 2,816), 
(d) {scene image, scene sketch} (training: 11,265, test: 2,816), and 
(e) sketch segmentation (training: 11,265, test: 2,816)


**EdgeGAN**

![](https://i.imgur.com/qBrbTLC.jpg)

It contains four sub-networks: two generators $G_I$ and $G_E$ , three discriminators $D_I$ , $D_E$ , and $D_J$ , an edge encoder $E$ and an image classifier $C$. EdgeGAN learns a joint embedding for an image and various-style edge maps depicting this image into a shared latent space where vectors can encode high-level attribute information from cross-modality data.

### **Proof**
- Datasets: SketchyCOCO
- Baselines: ContextualGAN, SketchyGAN, pix2pix; SPADE, Ashual ICCv19
- Metrics: FID, Shape Similarity, SSIM


### **Impact**

A weird task setting.


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

- [ICCV 2019: Image-to-Image Translation](https://arxivnote.ddlee.cn/2019/10/24/Image-to-Image-Translation-ICCV-2019.html)
- [Few-shot Video-to-Video Synthesis](https://arxivnote.ddlee.cn/2019/10/28/Few-Shot-Video-to-Video-Synthesis-NIPS.html)
- [Few-Shot Unsupervised Image-to-Image Translation](https://arxivnote.ddlee.cn/2019/10/27/Few-Shot-Unsupervised-Image-to-Image-Translation-ICCV.html)
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/2019/08/21/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE and INIT](https://arxivnote.ddlee.cn/2019/08/22/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [(DMIT)Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://arxivnote.ddlee.cn/2019/10/08/Multi-mappitng-Image-to-Image-Translation-Disentanglement.html)
- [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation - Junho Kim - 2019](https://arxivnote.ddlee.cn/2019/09/18/U-GAT-IT-Unsupervised-Generative-Attentional-Networks-with-Adaptive-Layer-Instance-Normalization-for-Image-to-Image-Translation-Junho-Kim-2019.html)
- [Towards Instance-level Image-to-Image Translation - Shen - CVPR 2019](https://arxivnote.ddlee.cn/2019/07/18/Towards-Instance-level-Image-to-Image-Translation-Shen-CVPR-2019.html)
