---
title: "MoCoGAN: Decomposing Motion and Content for Video Generation - Tulyakov - CVPR 2018"
tag:
- Video Generation
- GAN
---



## Info
- Title: **MoCoGAN: Decomposing Motion and Content for Video Generation**
- Task: **Video Generation**
- Author: Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz
- Date:  July 2017
- Arxiv: [1707.04993](https://arxiv.org/abs/1707.04993)
- Published: CVPR 2018

## Abstract
Visual signals in a video can be divided into content and motion. While content specifies which objects are in the video, motion describes their dynamics. Based on this prior, we propose the Motion and Content decomposed Generative Adversarial Network (MoCoGAN) framework for video generation. The proposed framework generates a video by mapping a sequence of random vectors to a sequence of video frames. Each random vector consists of a content part and a motion part. While the content part is kept fixed, the motion part is realized as a stochastic process. To learn motion and content decomposition in an unsupervised manner, we introduce a novel adversarial learning scheme utilizing both image and video discriminators. Extensive experimental results on several challenging datasets with qualitative and quantitative comparison to the state-of-the-art approaches, verify effectiveness of the proposed framework. In addition, we show that MoCoGAN allows one to generate videos with same content but different motion as well as videos with different content and same motion.

## Highlights & Drawbacks
- Propose a novel GAN framework for unconditional video generation, mapping noise vectors to videos.
- Show the proposed framework provides a means to control content and motion in video generation, which is absent in the existing video generation frameworks.


## Motivation & Design

![CleanShot 2019-09-21 at 20.51.57@2x](https://i.imgur.com/Obq90Fb.jpg)

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

The MoCoGAN framework for video generation. For a video, the content vector, $z_C$, is sampled once and fixed. Then, a series of random variables $[ε(1), ..., ε(K)]$ is sampled and mapped to a series of motion $M$ z(k)’s are from the recurrent neural network, $R_M$. A generator $G_I$ produces a frame,$ x ̃$ , using the content and the motion vectors ${zC, z(k)}$. The discriminators, DIM and DV, are trained on real and fake images and videos, respectively, sampled from the training set v and the generated set $v ̃$. The function S1 samples a single frame from a video, $S_T$ samples $T$ consequtive frames.


## Performance & Ablation Study

![CleanShot 2019-09-21 at 20.52.54@2x](https://i.imgur.com/ImIQwYX.jpg)
