---
title: "DLOW: Domain Flow for Adaptation and Generalization - Rui Gong - CVPR 2019"
tag:
- GAN
- Domain Transfer
redirect_from: /DLOW-Domain-Flow-for-Adaptation-and-Generalization-Rui-Gong-CVPR-2019.html
---



## Info

- Title: DLOW: Domain Flow for Adaptation and Generalization
- Task: Image Generation, Domain Adaptation
- Author: Rui Gong, Wen Li, Yuhua Chen, Luc Van Gool
- Arxiv: [1812.05418](https://arxiv.org/abs/1812.05418)
- Published: CVPR 2019

## Highlights
![DLOW: Domain Flow for Adaptation and Generalization - Rui Gong - CVPR 2019](https://i.imgur.com/rPOg6kQ.png)

The DLOW model is able to produce a sequence of intermediate domains shifting from the source domain to the target domain.

## Abstract

In this work, we present a domain flow generation(DLOW) model to bridge two different domains by generating a continuous sequence of intermediate domains flowing from one domain to the other. The benefits of our DLOW model are two-fold. First, it is able to transfer source images into different styles in the intermediate domains. The transferred images smoothly bridge the gap between source and target domains, thus easing the domain adaptation task. Second, when multiple target domains are provided for training, our DLOW model is also able to generate new styles of images that are unseen in the training data. We implement our DLOW model based on CycleGAN. A domainness variable is introduced to guide the model to generate the desired intermediate domain images. In the inference phase, a flow of various styles of images can be obtained by varying the domainness variable. We demonstrate the effectiveness of our model for both cross-domain semantic segmentation and the style generalization tasks on benchmark datasets. Our implementation is available at [https://github.com/ETHRuiGong/DLOW](https://github.com/ETHRuiGong/DLOW).

## Motivation & Design
![DLOW: Domain Flow for Adaptation and Generalization - Rui Gong - CVPR 2019](https://i.imgur.com/HDeVi1y.png)


The overview of our DLOW model: the generator takes domainness z as additional input to control the image translation and to reconstruct the source image; The domainness z is also used to weight the two discriminators.

The detailed network architecture.
![DLOW: Domain Flow for Adaptation and Generalization - Rui Gong - CVPR 2019](https://i.imgur.com/7o4OHpU.png)

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

## Experiments & Ablation Study
![DLOW: Domain Flow for Adaptation and Generalization - Rui Gong - CVPR 2019](https://i.imgur.com/4jjHqjv.png)


## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
