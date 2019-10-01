---
title: Semantics Disentangling for Text-to-Image Generation - Guojun Yin - CVPR 2019
tag:
- Text-to-Image
- GAN
redirect_from: /Semantics-Disentangling-for-Text-to-Image-Generation-Guojun-Yin-CVPR-2019.html
---



## Info
- Title: **Semantics Disentangling for Text-to-Image Generation**
- Task: Text-to-Image
- Author: Guojun Yin, Bin Liu, Lu Sheng, Nenghai Yu, Xiaogang Wang, Jing Shao
- Arxiv: [1904.01480](https://arxiv.org/abs/1904.01480)
- Published: CVPR 2019

## Highlights

Distill Semantic Commons from Text- The proposed SD-GAN distills semantic commons from the linguistic descriptions, based on which the generated images can keep generation consistency under expression variants. To our best knowledge, it is the first time to introduce the Siamese mechanism into the cross-modality generation.

Retain Semantic Diversities & Details from Text- To complement the Siamese mechanism that may lose unique semantic diversities, we design an enhanced visual-semantic embedding method by reformulating the batch normalization layer with the instance linguistic cues. The linguistic embedding can further guide the visual pattern synthesis for fine-grained image generation.

## Abstract

Synthesizing photo-realistic images from text descriptions is a challenging problem. Previous studies have shown remarkable progresses on visual quality of the generated images. In this paper, we consider semantics from the input text descriptions in helping render photo-realistic images. However, diverse linguistic expressions pose challenges in extracting consistent semantics even they depict the same thing. To this end, we propose a novel photo-realistic text-to-image generation model that implicitly disentangles semantics to both fulfill the high-level semantic consistency and low-level semantic diversity. To be specific, we design (1) a Siamese mechanism in the discriminator to learn consistent high-level semantics, and (2) a visual-semantic embedding strategy by semantic-conditioned batch normalization to find diverse low-level semantics. Extensive experiments and ablation studies on CUB and MS-COCO datasets demonstrate the superiority of the proposed method in comparison to state-of-the-art methods.

## Motivation & Design

The architecture of SD-GAN. The robust semantic-related text-to-image generation is optimized by contrastive losses based on a Siamese structure. The Semantic-Conditioned Batch Normalization (SCBN) is introduced to further retain the unique semantic diversities from text and embed the visual features modulated to the textual cues.
![Semantics Disentangling for Text-to-Image Generation - Guojun Yin - CVPR 2019](https://i.imgur.com/LQsmfRb.png)

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
### The Generators in the proposed SD-GAN

(a) G0 , the generator at the initial stage from the linguistic to vision, (b) G1/G2, the generator at the second/third stage for generating higher-resolution images based on generated visual features
at the former stage. The SCBNs operate at the end of each upsampling layer.
![Semantics Disentangling for Text-to-Image Generation - Guojun Yin - CVPR 2019](https://i.imgur.com/kQHvjG4.png)

###Discriminators
![Semantics Disentangling for Text-to-Image Generation - Guojun Yin - CVPR 2019](https://i.imgur.com/sNZ2Ks1.png)

###Contrastive Loss
![Semantics Disentangling for Text-to-Image Generation - Guojun Yin - CVPR 2019](https://i.imgur.com/LDP8jZV.png)

###Semantic-conditioned Batch Normalization(SCBN)

![Semantics Disentangling for Text-to-Image Generation - Guojun Yin - CVPR 2019](https://i.imgur.com/K7dIOID.png)

![Semantics Disentangling for Text-to-Image Generation - Guojun Yin - CVPR 2019](https://i.imgur.com/TvAFcm8.png)

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

(1) sentence-level cues that consists of a one-hidden-layer MLP to extract modulation parameters from the sentence feature vector; and (2) word-level cues that uses VSE module to fuse the visual
features and word features. Note that the illustration only takes γc as the example and the implementation for βc is alike.

## Experiments & Ablation Study
![Semantics Disentangling for Text-to-Image Generation - Guojun Yin - CVPR 2019](https://i.imgur.com/vmbaAXP.png)

## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
