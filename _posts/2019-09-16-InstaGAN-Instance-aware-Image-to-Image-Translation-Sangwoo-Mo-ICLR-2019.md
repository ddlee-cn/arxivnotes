---
title: "InstaGAN: Instance-aware Image-to-Image Translation - Sangwoo Mo - ICLR 2019"
tag:
- Image-to-Image Translation
- GAN
---



## Info
- Title: **InstaGAN: Instance-aware Image-to-Image Translation**
- Task: **Image-to-Image Translation**
- Author: Sangwoo Mo, Minsu Cho, Jinwoo Shin
- Date:  Dec. 2018
- Arxiv: [1812.10889](https://arxiv.org/abs/1812.10889)
- Published: ICLR 2019

## Highlights & Drawbacks
- Instance-level translation with semantic map
- Sequential mini-batch training strategy

## Abstract
Unsupervised image-to-image translation has gained considerable attention due to the recent impressive progress based on generative adversarial networks (GANs). However, previous methods often fail in challenging cases, in particular, when an image has multiple target instances and a translation task involves significant changes in shape, e.g., translating pants to skirts in fashion images. To tackle the issues, we propose a novel method, coined instance-aware GAN (InstaGAN), that incorporates the instance information (e.g., object segmentation masks) and improves multi-instance transfiguration. The proposed method translates both an image and the corresponding set of instance attributes while maintaining the permutation invariance property of the instances. To this end, we introduce a context preserving loss that encourages the network to learn the identity function outside of target instances. We also propose a sequential mini-batch inference/training technique that handles multiple instances with a limited GPU memory and enhances the network to generalize better for multiple instances. Our comparative evaluation demonstrates the effectiveness of the proposed method on different image datasets, in particular, in the aforementioned challenging cases.


## Motivation & Design
![CleanShot 2019-09-22 at 11.07.45@2x](https://i.imgur.com/LnP6Vx5.jpg)

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

(a) Overview of InstaGAN, where generators GXY, GYX and discriminator DX, DY follows the architectures in (b) and (c), respectively. Each network is designed to encode both an image and set of instance masks. G is permutation equivariant, and D is permutation invariant to the set order. To achieve properties, we sum features of all set elements for invariance, and then concatenate it with the identity mapping for equivariance.


![CleanShot 2019-09-22 at 11.06.55@2x](https://i.imgur.com/wPsmqRC.jpg)

Overview of the sequential mini-batch training with instance subsets (mini-batches) of size 1,2, and 1, as shown in the top right side. The content loss is applied to the intermediate samples of current mini-batch, and GAN loss is applied to the samples of aggregated mini-batches. We detach every iteration in training, in that the real line indicates the backpropagated paths and dashed lines indicates the detached paths. See text for details.

## Performance & Ablation Study

![CleanShot 2019-09-22 at 11.09.59@2x](https://i.imgur.com/8sdR7Fg.jpg)


![CleanShot 2019-09-22 at 11.09.04@2x](https://i.imgur.com/8oqUKQC.jpg)

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

Ablation study on the effect of each component of our method: the InstaGAN architecture, the context preserving loss, and the sequential mini-batch inference/training algorithm, which are denoted as InstaGAN, Lctx, and Sequential, respectively.

![CleanShot 2019-09-22 at 11.09.21@2x](https://i.imgur.com/jmmycKj.jpg)


Ablation study on the effects of the sequential mini-batch inference/training technique. The left and right side of title indicates which method used for training and inference, respectively, where “One” and “Seq” indicate the one-step and sequential schemes, respectively.


## Code
- [PyTorch](https://github.com/sangwoomo/instagan)


## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
