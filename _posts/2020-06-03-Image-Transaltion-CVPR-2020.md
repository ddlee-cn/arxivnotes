---
title: "CVPR 2020: Image-to-Image Translation(1)"
tag: 
- Image-to-Image Translation
- GAN
---

## SEAN: Image Synthesis with Semantic Region-Adaptive Normalization
- Author: Peihao Zhu, Rameen Abdal, Yipeng Qin, Peter Wonka
- Arxiv: [1911.12861](https://arxiv.org/abs/1911.12861.pdf)
- [GitHub](https://github.com/ZPdesu/SEAN)


### **Problem**
synthetic image generation

### **Assumption in prior work**

Starting from SPADE, 1) use only one style code for whole image, 2) insert style code only in the beginning of network. 

None of previous networks use style information to generate spatially varying normalization parameters.

### **Insight**

![](https://i.imgur.com/SXcEANr.jpg)

control the style of each semantic region individually, e.g., we can specify one style reference image per region

use style input images to create spatially varying normalization parameters per semantic region. An important aspect of this work is that the spatially varying normalization parameters are dependent on the segmentation mask as well as the style input images.


### **Technical overview**

**SEAN normalization**

![](https://i.imgur.com/OYB9AXk.jpg)

The input are style matrix ST and segmentation mask M. In the upper part, the style codes in ST undergo a per style convolution and are then broadcast to their corresponding regions according to M to yield a style map. The style map is processed by conv layers to produce per pixel normalization values $\gamma^s$ and $\beta^s$ . The lower part (light blue layers) creates per pixel normalization values using only the region information similar to SPADE.

**The Generator**

![](https://i.imgur.com/agniyxT.jpg)
(A) On the left, the style encoder takes an input image and outputs a style matrix ST. The generator on the right consists of interleaved SEAN ResBlocks and Upsampling layers. (B) A detailed view of a SEAN ResBlock used in (A).

### **Proof**
- Datasets: ADE20k, cityscapes, CelebA-HQ, Facades
- Baselines: pix2pixHD, SPADE
- Metrics: mIoU, pixel accuracy, FID; SSIM, RMSE, PSNR(for reconstruction)


![](https://i.imgur.com/dxevdI2.jpg)

![](https://i.imgur.com/9AJGFPN.jpg)



### **Impact**
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

### **Problem**
Conditional Image Synthesis


### **Assumption in prior work**
Traditional convolution-based generative adversarial networks synthesize images based on hierarchical local operations, where long-range dependency relation is implicitly modeled with a Markov chain. It is still not sufficient for categories with complicated structures.

Self-Attention GAN: the self-attention module requires computing the correlation between every two points in the feature map. Therefore, the computational cost grows rapidly as the feature map becomes large.

Instance Normalization (IN): the previous solution of (IN) normalizes the mean and variance of a feature map along its spatial dimensions. This strategy ignores the fact that different locations may correspond to semantics with varying mean and variance.

### **Insight**
Attentive Normalization (AN) predicts a semantic layout from the input feature map and then conduct regional instance normalization on the feature map based on this layout.


### **Technical overview**

![](https://i.imgur.com/vNU3vj3.jpg)

AN is formed by the proposed semantic layout learning (SLL) module, and a regional normalization, as shown in Figure 2. It has a semantics learning branch and a self-sampling branch. The semantic learning branch employs a certain number of convolutional filters to capture regions with different semantics (which are activated by a specific filter), with the assumption that each filter in this branch corresponds to some semantic entities.

### **Proof**
- Datasets: ImageNet; Paris Streetview
- Baselines: SN-GAN, SA-GAN, BigGAN (Conditional Synthesis); CA (inpainting)
- Metrics: FID, IS (Conditional Synthesis); PSRN, SSIM (inpainting)

### **Impact**
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

### **Problem**
an image-to-image translation problem suitable for the setting when domain labels are unavailable.


### **Assumption in prior work**
Image-to-image translation approaches require domain labels at training as well as at inference time. The recent FUNIT model relaxes this constraint partially. Thus, to extract the style at inference time, it uses several images from the target domain as guidance for translation (known as the few-shot setting). The domain annotations are however still needed during training.


### **Insight**

The only external (weak) supervision used by our approach are coarse segmentation maps estimated using an off-the-shelf semantic segmentation network.

### **Technical overview**

![](https://i.imgur.com/krRYkTz.jpg)

HiDT learning data flow. We show half of the (symmetric) architecture; s′ = Es(x′) is the style extracted from the other image x′, and ŝ′ is obtained similarly to ŝ with x and x′ swapped. Light blue nodes denote data elements; light green, loss functions; others, functions (subnetworks). Functions with identical labels have shared weights. Adversarial losses are omitted for clarity.

![](https://i.imgur.com/EGDGxE6.jpg)

Enhancement scheme: the input is split into subimages (color-coded) that are translated individually by HiDT at medium resolution. The outputs are then merged using the merging network Genh. For illustration purposes, we show upsampling by a factor of two, but in the experiments we use a factor of four. We also apply bilinear downsampling (with shifts – see text for detail) rather than strided subsampling when decomposing the input into medium resolution images


### **Proof**
- Datasets: 20,000 landscape photos labeled by a pre-trained classifier
- Baselines: FUNIT, DRIT++
- Metrics: domain-invariant perceptual distance (DIPD), adapted IS, 


### **Impact**

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


### **Problem**
Unsupervised image-to-image translation

### **Assumption in prior work**
Current translation frameworks will abandon the discriminator once the training process is completed.
This paper contends a novel role of the discriminator by reusing it for encoding the images of the target domain.


### **Insight**
We reuse early layers of certain number in the discriminator as the encoder of the target domain

We develop a decoupled training strategy by which the encoder is only trained when maximizing the adversary loss while keeping frozen otherwise.


### **Technical overview**

![](https://i.imgur.com/TCWSl99.jpg)


### **Proof**

- Dataset: horse↔zebra, summer↔winter, vangogh↔photo and cat↔dog
- Baselines: CycleGAN, UNIT, MUNIT, DRIT, U-GAT-IT
- Metrics: FID, KID

![](https://i.imgur.com/tovjzTM.jpg)


### **Impact**

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


### **Problem**
Few-shot(both in source and target) unpaired image-to-image translation

![](https://i.imgur.com/47KLhbV.jpg)

(c) Few-shot semi-supervised (Ours): same as few-shot, but the source domain has only a limited amount of labeled data at train time.

### **Assumption in prior work**
First, the target domain is required to contain the same categories or attributes as the source domain at test time, therefore failing to scale to unseen categories (see Fig. 1(a)). Second, they highly rely upon having access to vast quantities of labeled data (Fig. 1(a, b)) at train time. Such labels provide useful information during the training process and play a key role in some settings (e.g. scalable I2I translation).


### **Insight**

We propose using semi-supervised learning to reduce the requirement of labeled source images and effectively use unlabeled data. More concretely, we assign pseudo-labels to the unlabeled images based on an initial small set of labeled images. These pseudo-labels provide soft supervision to train an image translation model from source images to unseen target domains. Since this mechanism can potentially introduce noisy labels, we employ a pseudo-labeling technique that is highly robust to noisy labels. In order to further leverage the unlabeled images from the dataset (or even external images), we use a cycle consistency constraint [48].

### **Technical overview**
![](https://i.imgur.com/puJzUMJ.jpg)



### **Proof**
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
