---
title: Image-to-Image Translation at CVPR 2020
tag: 
- Imaget-to-Image Translation
- GAN
---

## (CoCosNet) Cross-domain Correspondence Learning for Exemplar-based Image Translation

- Author: Pan Zhang, Bo Zhang, Dong Chen, Lu Yuan, Fang Wen
- Arxiv: [2004.05571](https://arxiv.org/abs/2004.05571)
- [Project Site](https://panzhang0212.github.io/CoCosNet/)


### **Problem**: What problem is it solving? Why does this problem matter?
exemplar-based image translation


### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?

Previous exemplar-based method only use style code globally.
The style code only characterizes the global style of the exemplar, regardless of spatial relevant information. Thus, it causes some local style “wash away” in the ultimate image.

Deep Image Analogy is not cross-domain, may fail to handle a more challenging mapping from mask (or edge, keypoints) to photo since the pretrained network does not recognize such images.

### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?

![](https://i.imgur.com/EOLx82w.jpg)


With the cross-domain correspondence, we present a general solution to exemplar-based image translation, that for the first time, outputs images resembling the fine structures of the exemplar at instance level.

### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?

![](https://i.imgur.com/FtdEk3G.jpg)


The network architecture comprises two sub-networks: 1) Cross-domain correspondence Network transforms the inputs from distinct domains to an intermediate feature domain where reliable dense correspondence can be established; 2) Translation network, employs a set of spatially-variant de-normalization blocks [38] to progressively synthesizes the output, using the style details from a warped exemplar which is semantically aligned to the mask (or edge, keypoints map) according to the estimated correspondence.


### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?

- Datasets: ADE20k, CelebA-HQ, Deepfashion
- Baselines: Pix2pixHD, SPADE, MUNIT, SIMS, EGSC-IT
- Metrics: 1)FID, SWD; 2)semantic consistency: high-level feature distance from ImageNet VGG; 3)style relevance: low-level feature distance from ImageNet VGG 4)User Study ranking

![](https://i.imgur.com/IAk984j.jpg)


### **Impact**: What are the implications of this paper? How will it change how we think about the problem?

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


### **Problem**: What problem is it solving? Why does this problem matter?
multiple domain image translation


### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?
Existing Image-to-image translation methods have only considered a mapping between two domains, they are not scalable to the increasing number of domains. For example, having K domains, these methods require to train K(K-1) generators to handle translations between each and every domain, limiting their practical usage.

StarGAN still learns a deterministic mapping per each domain, which does not capture the multi-modal nature of the data distribution. 


### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?

![](https://i.imgur.com/BkHYRKw.jpg)

generate diverse images across multiple domains.


### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?

![](https://i.imgur.com/I1yp3uT.jpg)

(a) The generator translates an input image into an output image reflecting the domain-specific style code. (b) The mapping network transforms a latent code into style codes for multiple domains, one of which is randomly selected during training. (c) The style encoder extracts the style code of an image, allowing the generator to perform reference-guided image synthesis. (d) The discriminator distinguishes between real and fake images from multiple domains

In particular, we start from StarGAN and replace its domain label with our proposed domain-specific style code that can represent diverse styles of a specific domain. To this end, we introduce two modules, a mapping network and a style encoder. The mapping network learns to transform random Gaussian noise into a style code, while the encoder learns to extract the style code from a given reference image. Considering multiple domains, both modules have multiple output branches, each of which provides style codes for a specific domain. Finally, utilizing these style codes, our generator learns to successfully synthesize diverse images over multiple domains

### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?
- Datasets: CelebA-HQ, AFHQ(new)
- Baselines: MUNIT, DRIT, MSGAN, StarGAN
- Metrics: FID, LPIPS

![](https://i.imgur.com/qL2aX2s.jpg)


### **Impact**: What are the implications of this paper? How will it change how we think about the problem?

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

- Author: 
- Arxiv:
- [Project Site]
- [GitHub]


### **Problem**: What problem is it solving? Why does this problem matter?



### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?



### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?



### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?



### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?



### **Impact**: What are the implications of this paper? How will it change how we think about the problem?


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


### **Problem**: What problem is it solving? Why does this problem matter?
from panoptic map to image


### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?
Previous conditional image synthesis algorithms mostly rely on semantic maps, and often fail in complex environments where multiple instances occlude each other.
This is the result of conventional convolution and upsampling algorithms being independent of class and instance boundaries.

### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?
We are interested in panoptic maps because semantic maps do not provide sufficient information to synthesize “things” (instances) especially in complex environments with multiple of them interacting with each other.

We propose Panoptic aware upsampling that addresses the misalignment between the upsampled low resolution features and high resolution panoptic maps. This ensures that the semantic and instance details are not lost, and that we also maintain higher accuracy alignment between the generated images and the panoptic maps.

![](https://i.imgur.com/J9JZ0dP.jpg)


### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?

![](https://i.imgur.com/I1oHWBQ.jpg)

**Panoptic Aware Convolution Layer**
![](https://i.imgur.com/wPZumt0.jpg)

Panoptic aware partial convolution layer takes a panoptic map (colorized for visualization) and based on the center of each sliding window it generates a binary mask, M. The pixels that share the same identity with the center of the window are assigned 1 and the others 0.

**Panoptic Aware Upsampling Layer**

![](https://i.imgur.com/LHaP3qk.jpg)

As shown in Figure (top), first we correct misalignment by replicating a feature vector from a neighboring pixel that belongs to the same panoptic instance. This operation is different from nearest neighbor upsampling which would always replicate the top-left feature. Second, as shown in Figure (bottom), we resolve pixels where new semantic or instance classes have just appeared by encoding new features from semantic maps with Panoptic aware convolution layer.

### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?
- Datasets: COCO-Stuff, Cityscapes
- Baselines: CRN, SIMS, SPADE
- Metrics: detAP(for instance detection), mIoU, pixel Acc, FID



![](https://i.imgur.com/QK1Z4mw.jpg)



### **Impact**: What are the implications of this paper? How will it change how we think about the problem?


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

### **Problem**: What problem is it solving? Why does this problem matter?

![](https://i.imgur.com/5lIUpcu.jpg)

controllably generating realistic images with many objects and relationships from a freehand scene-level sketch


### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?
The author argue that this is a new problem.


### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?
Using freehand sketches as conditional signal is hard.


### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?

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

### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?
- Datasets: SketchyCOCO
- Baselines: ContextualGAN, SketchyGAN, pix2pix; SPADE, Ashual ICCv19
- Metrics: FID, Shape Similarity, SSIM


### **Impact**: What are the implications of this paper? How will it change how we think about the problem?

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

## SEAN: Image Synthesis with Semantic Region-Adaptive Normalization
- Author: Peihao Zhu, Rameen Abdal, Yipeng Qin, Peter Wonka
- Arxiv: [1911.12861](https://arxiv.org/abs/1911.12861.pdf)
- [GitHub](https://github.com/ZPdesu/SEAN)


### **Problem**: What problem is it solving? Why does this problem matter?
synthetic image generation

### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?

Starting from SPADE, 1) use only one style code for whole image, 2) insert style code only in the beginning of network. 

None of previous networks use style information to generate spatially varying normalization parameters.

### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?

![](https://i.imgur.com/SXcEANr.jpg)

control the style of each semantic region individually, e.g., we can specify one style reference image per region

use style input images to create spatially varying normalization parameters per semantic region. An important aspect of this work is that the spatially varying normalization parameters are dependent on the segmentation mask as well as the style input images.


### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?

**SEAN normalization**

![](https://i.imgur.com/OYB9AXk.jpg)

The input are style matrix ST and segmentation mask M. In the upper part, the style codes in ST undergo a per style convolution and are then broadcast to their corresponding regions according to M to yield a style map. The style map is processed by conv layers to produce per pixel normalization values $\gamma^s$ and $\beta^s$ . The lower part (light blue layers) creates per pixel normalization values using only the region information similar to SPADE.

**The Generator**

![](https://i.imgur.com/agniyxT.jpg)
(A) On the left, the style encoder takes an input image and outputs a style matrix ST. The generator on the right consists of interleaved SEAN ResBlocks and Upsampling layers. (B) A detailed view of a SEAN ResBlock used in (A).

### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?
- Datasets: ADE20k, cityscapes, CelebA-HQ, Facades
- Baselines: pix2pixHD, SPADE
- Metrics: mIoU, pixel accuracy, FID; SSIM, RMSE, PSNR(for reconstruction)


![](https://i.imgur.com/dxevdI2.jpg)

![](https://i.imgur.com/9AJGFPN.jpg)



### **Impact**: What are the implications of this paper? How will it change how we think about the problem?
application: style interpolation
an per-region extension to SPADE

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


## Attentive Normalization for Conditional Image Generation
- Author: Yi Wang, Yubei Chen, Xiangyu Zhang, Jian-Tao Sun, Jiaya Jia less
- Arxiv: [2004.03828](https://arxiv.org/abs/2004.03828.pdf)


![](https://i.imgur.com/XxGDieo.jpg)
Conditional image generation of a GAN framework using our proposed attentive normalization module. (a) Class conditional image generation. (b) Image inpainting.

### **Problem**: What problem is it solving? Why does this problem matter?
Conditional Image Synthesis


### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?
Traditional convolution-based generative adversarial networks synthesize images based on hierarchical local operations, where long-range dependency relation is implicitly modeled with a Markov chain. It is still not sufficient for categories with complicated structures.

Self-Attention GAN: the self-attention module requires computing the correlation between every two points in the feature map. Therefore, the computational cost grows rapidly as the feature map becomes large.

Instance Normalization (IN): the previous solution of (IN) normalizes the mean and variance of a feature map along its spatial dimensions. This strategy ignores the fact that different locations may correspond to semantics with varying mean and variance.

### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?
Attentive Normalization (AN) predicts a semantic layout from the input feature map and then conduct regional instance normalization on the feature map based on this layout.


### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?

![](https://i.imgur.com/vNU3vj3.jpg)

AN is formed by the proposed semantic layout learning (SLL) module, and a regional normalization, as shown in Figure 2. It has a semantics learning branch and a self-sampling branch. The semantic learning branch employs a certain number of convolutional filters to capture regions with different semantics (which are activated by a specific filter), with the assumption that each filter in this branch corresponds to some semantic entities.

### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?
- Datasets: ImageNet; Paris Streetview
- Baselines: SN-GAN, SA-GAN, BigGAN (Conditional Synthesis); CA (inpainting)
- Metrics: FID, IS (Conditional Synthesis); PSRN, SSIM (inpainting)

### **Impact**: What are the implications of this paper? How will it change how we think about the problem?
semantics-aware attention + regional normalization


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



## High-Resolution Daytime Translation Without Domain Labels

- Author: Ivan Anokhin, Pavel Solovev, Denis Korzhenkov, Alexey Kharlamov, Taras Khakhulin, Alexey Silvestrov, Sergey I. Nikolenko, Victor S. Lempitsky, Gleb Sterkin 
- Arxiv: [2003.08791](https://arxiv.org/abs/2003.08791.pdf)
- [GitHub](https://github.com/saic-mdal/HiDT)
- [Project Site](https://saic-mdal.github.io/HiDT/)

### **Problem**: What problem is it solving? Why does this problem matter?
an image-to-image translation problem suitable for the setting when domain labels are unavailable.


### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?
Image-to-image translation approaches require domain labels at training as well as at inference time. The recent FUNIT model relaxes this constraint partially. Thus, to extract the style at inference time, it uses several images from the target domain as guidance for translation (known as the few-shot setting). The domain annotations are however still needed during training.


### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?

The only external (weak) supervision used by our approach are coarse segmentation maps estimated using an off-the-shelf semantic segmentation network.

### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?

![](https://i.imgur.com/krRYkTz.jpg)

HiDT learning data flow. We show half of the (symmetric) architecture; s′ = Es(x′) is the style extracted from the other image x′, and ŝ′ is obtained similarly to ŝ with x and x′ swapped. Light blue nodes denote data elements; light green, loss functions; others, functions (subnetworks). Functions with identical labels have shared weights. Adversarial losses are omitted for clarity.

![](https://i.imgur.com/EGDGxE6.jpg)

Enhancement scheme: the input is split into subimages (color-coded) that are translated individually by HiDT at medium resolution. The outputs are then merged using the merging network Genh. For illustration purposes, we show upsampling by a factor of two, but in the experiments we use a factor of four. We also apply bilinear downsampling (with shifts – see text for detail) rather than strided subsampling when decomposing the input into medium resolution images


### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?
- Datasets: 20,000 landscape photos labeled by a pre-trained classifier
- Baselines: FUNIT, DRIT++
- Metrics: domain-invariant perceptual distance (DIPD), adapted IS, 


### **Impact**: What are the implications of this paper? How will it change how we think about the problem?

High-resolution translation

![](https://i.imgur.com/0PmfIA0.jpg)

Swapping styles between two images. Original images are shown on the main diagonal. The examples show that HiDT is capable to swap the styles between two real images while preserving details. 


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


## Reusing Discriminators for Encoding: Towards Unsupervised Image-to-Image Translation

- Author: Runfa Chen, Wenbing Huang, Binghui Huang, Fuchun Sun ∗ , Bin Fang
- Arxiv: [2003.00273](https://arxiv.org/abs/2003.00273.pdf)
- [GitHub](https://github.com/alpc91/NICE-GAN-pytorch)
- [Project Site](https://alpc91.github.io/NICE-GAN-pytorch)

![](https://i.imgur.com/rMwBsXB.jpg)


### **Problem**: What problem is it solving? Why does this problem matter?
Unsupervised image-to-image translation

### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?
Current translation frameworks will abandon the discriminator once the training process is completed.
This paper contends a novel role of the discriminator by reusing it for encoding the images of the target domain.


### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?
We reuse early layers of certain number in the discriminator as the encoder of the target domain

We develop a decoupled training strategy by which the encoder is only trained when maximizing the adversary loss while keeping frozen otherwise.


### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?

![](https://i.imgur.com/TCWSl99.jpg)


### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?

- Dataset: horse↔zebra, summer↔winter, vangogh↔photo and cat↔dog
- Baselines: CycleGAN, UNIT, MUNIT, DRIT, U-GAT-IT
- Metrics: FID, KID

![](https://i.imgur.com/tovjzTM.jpg)


### **Impact**: What are the implications of this paper? How will it change how we think about the problem?

sounds like a plug-in strategy to all I2I frameworks.

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


## Semi-supervised Learning for Few-shot Image-to-Image Translation


- Author: Yaxing Wang, Salman Khan, Abel Gonzalez-Garcia, Joost van de Weijer, Fahad Shahbaz Khan
- Arxiv: [2003.13853](https://arxiv.org/abs/2003.13853.pdf)
- [GitHub](https://github.com/yaxingwang/SEMIT)


### **Problem**: What problem is it solving? Why does this problem matter?
Few-shot(both in source and target) unpaired image-to-image translation

![](https://i.imgur.com/47KLhbV.jpg)

(c) Few-shot semi-supervised (Ours): same as few-shot, but the source domain has only a limited amount of labeled data at train time.

### **Assumption in prior work**: What was the assumption that prior research made when solving this problem? Why was that assumption inadequate?
First, the target domain is required to contain the same categories or attributes as the source domain at test time, therefore failing to scale to unseen categories (see Fig. 1(a)). Second, they highly rely upon having access to vast quantities of labeled data (Fig. 1(a, b)) at train time. Such labels provide useful information during the training process and play a key role in some settings (e.g. scalable I2I translation).


### **Insight**: What is the novel idea that this paper introduces, breaking from that prior assumption?

We propose using semi-supervised learning to reduce the requirement of labeled source images and effectively use unlabeled data. More concretely, we assign pseudo-labels to the unlabeled images based on an initial small set of labeled images. These pseudo-labels provide soft supervision to train an image translation model from source images to unseen target domains. Since this mechanism can potentially introduce noisy labels, we employ a pseudo-labeling technique that is highly robust to noisy labels. In order to further leverage the unlabeled images from the dataset (or even external images), we use a cycle consistency constraint [48].

### **Technical overview**: How did the paper implement that insight? I.e., What did they build, or what did they prove, and how?
![](https://i.imgur.com/puJzUMJ.jpg)



### **Proof**: How did the paper evaluate or prove that its insight is correct, and is better than holding on to the old assumption?
- Metrics: FID, IS
- Baselines: CycleGAN, StarGAN, MUIT, FUNIT



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
