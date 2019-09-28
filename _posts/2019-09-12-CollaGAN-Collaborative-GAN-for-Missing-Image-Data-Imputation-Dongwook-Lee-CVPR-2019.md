---
title: "CollaGAN : Collaborative GAN for Missing Image Data Imputation - Dongwook Lee - CVPR 2019"
tag:
- Image Generation
- GAN
---



## Info

- Title: **CollaGAN : Collaborative GAN for Missing Image Data Imputation**
- Task: Image Generation
- Author: Dongwook Lee 
- Arxiv: [1829](https://arxiv.org/abs/)
- Published: CVPR 2019

## Highlights

Many-to-One image generation: The underlying image manifold can be learned more synergistically from the multiple input data set sharing the same manifold structure, rather than from a single input. Therefore, the estimation of missing data using CollaGAN is more accurate.

## Abstract

In many applications requiring multiple inputs to obtain a desired output, if any of the input data is missing, it often introduces large amounts of bias. Although many techniques have been developed for imputing missing data, the image imputation is still difficult due to complicated nature of natural images. To address this problem, here we proposed a novel framework for missing image data imputation, called Collaborative Generative Adversarial Network (CollaGAN). CollaGAN converts an image imputation problem to a multi-domain images-to-image translation task so that a single generator and discriminator network can successfully estimate the missing data using the remaining clean data set. We demonstrate that CollaGAN produces the images with a higher visual quality compared to the existing competing approaches in various image imputation tasks.s

## Motivation & Design
![](https://i.imgur.com/thgNWJO.png)

 Image translation tasks using (a) cross-domain models, (b) StarGAN, and (c) the proposed collaborative GAN(CollaGAN). Cross-domain model needs large number of generators to handle multi-class data. StarGAN and CollaGAN use a single generator with one input and multiple inputs, respectively, to synthesize the target domain image.

![](https://i.imgur.com/NexoGSb.png)

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

D has two branches: domain classification Dclsf and source classficiation Dgan (real/fake). First, Dclsf is only trained by (1) the loss calculated from real samples (left). Then G reconstructs the target domain image using the set of input images (middle). For the cycle consistency, the generated fake image re-entered to the G with inputs images and G produces the multiple reconstructed outputs in original domains. Here, Dclsf and Dgan are simultaneously trained by the loss from only (1) real images and both (1) real & (2) fake images, respectively (right).

## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
