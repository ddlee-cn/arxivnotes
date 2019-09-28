---
title: Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping - Huan Fu - CVPR 2019
tag:
- GAN
- Domain Transfer
---




## Info

- Title: **Geometry-Consistent Generative Adversarial Networks for One-Sided Unsupervised Domain Mapping**
- Task: Image Generation
- Author:  Chaohui Wang, Huan Fu, Mingming Gong, Kayhan Batmanghelich, Kun Zhang, Dacheng Tao
- Arxiv: [1809.05852](https://arxiv.org/abs/1809.05852)
- Published: CVPR 2019

## Abstract

Unsupervised domain mapping aims to learn a function to translate domain X to Y by a function GXY in the absence of paired examples. Finding the optimal GXY without paired data is an ill-posed problem, so appropriate constraints are required to obtain reasonable solutions. One of the most prominent constraints is cycle consistency, which enforces the translated image by GXY to be translated back to the input image by an inverse mapping GYX. While cycle consistency requires the simultaneous training of GXY and GY X, recent studies have shown that one-sided domain mapping can be achieved by preserving pairwise distances between images. Although cycle consistency and distance preservation successfully constrain the solution space, they overlook the special properties that simple geometric transformations do not change the semantic structure of images. Based on this special property, we develop a geometry-consistent generative adversarial network (GcGAN), which enables one-sided unsupervised domain mapping. GcGAN takes the original image and its counterpart image transformed by a predefined geometric transformation as inputs and generates two images in the new domain coupled with the corresponding geometry-consistency constraint. The geometry-consistency constraint reduces the space of possible solutions while keep the correct solutions in the search space. Quantitative and qualitative comparisons with the baseline (GAN alone) and the state-of-the-art methods including CycleGAN and DistanceGAN demonstrate the effectiveness of our method.

## Motivation & Design

![](https://i.imgur.com/5BOZ9h8.png)

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
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
