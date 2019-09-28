---
title: "DeSIGN: Design Inspiration from Generative Networks - Sabi - 2018"
tag:
- GAN
- Application
redirect_from: /DeSIGN-Design-Inspiration-from-Generative-Networks-Sabi-2018.html
---

## Info
- Title: **DeSIGN: Design Inspiration from Generative Networks**
- Author: O. Sbai, M. Elhoseiny, A. Bordes, Y. LeCun, and C. Couprie
- Date: Apr. 2018
- Arxiv: [1804.00921](https://arxiv.org/abs/1804.00921)

## Highlights & Drawbacks
(i) different Generative Adversarial Networks architectures that start from noise vectors to generate fashion items
(ii) novel loss functions that encourage creativity, inspired from Sharma-Mittal divergence, a generalized mutual information measure for the widely used relative entropies such as Kullback-Leibler
(iii) a generation process following the key elements of fashion design (disentangling shape and texture components)


<!-- more -->

## Motivation & Design
**Loss Design**
*Binary cross entropy loss (CAN (Elgammal et al., 2017))*
Given the adversarial network’s branch Dc trained to classify different textures or shapes, we can use the CAN loss LCAN as LG creativity to create a new style that confuses Db:

$$
\mathcal{L}_{\mathrm{CAN}}=-\sum_{i} \sum_{k=1}^{K} \frac{1}{K} \log \left(\sigma\left(D_{b, k}\left(G\left(z_{i}\right)\right)\right)\right)+\frac{K-1}{K} \log \left(1-\sigma\left(D_{b, k}\left(G\left(z_{i}\right)\right)\right)\right)
$$

*Multi-class Cross Entropy loss*
We propose to use as LG creativity the Multi-class Cross Entropy (MCE) loss between the class prediction of the discriminator and the uniform distribution. The goal is for the generator to make the generations hard to classify by the discriminator.

$$
\mathcal{L}_{\mathrm{MCE}}=-\sum_{i} \sum_{k=1}^{K} \frac{1}{K} \log \left(\frac{e^{D_{b, k}\left(G\left(z_{i}\right)\right)}}{\sum_{q=1}^{K} e^{D_{b, q}\left(G\left(z_{i}\right)\right)}}\right)=-\sum_{i} \sum_{k=1}^{K} \frac{1}{K} \log \left(\hat{D}_{i}\right)
$$

*Generalized Sharma-Mittal Divergence*

$$
S M(\alpha, \beta)(p \| q)=\frac{1}{\beta-1}\left[\sum_{i}\left(p_{i}^{1-\alpha} q_{i}^{\alpha}\right)^{\frac{1-\beta}{1-\alpha}}-1\right]
$$

Each of the Re ́nyi, Tsallis and Kullback-Leibler (KL) divergences can be defined as limiting cases of SM divergence as follows:

$$
\begin{aligned} R_{\alpha}(p \| q) &=\lim _{\beta \rightarrow 1} S M_{\alpha, \beta}(p \| q)=\frac{1}{\alpha-1} \ln \left(\sum_{i} p_{i}^{\alpha} q_{i}^{1-\alpha}\right) ) \\ T_{\alpha}(p \| q) &=\lim _{\beta \rightarrow \alpha} S M_{\alpha, \beta}(p \| q)=\frac{1}{\alpha-1}\left(\sum_{i} p_{i}^{\alpha} q_{i}^{1-\alpha}\right)-1 ) \\ K L(p \| q) &=\lim _{\beta \rightarrow 1, \alpha \rightarrow 1} S M_{\alpha, \beta}(p \| q)=\sum_{i} p_{i} \ln \left(\frac{p_{i}}{q_{i}}\right) \end{aligned}
$$


**Various Architecture**
- DCGAN
- Unconditioned StackGAN
- StyleGAN conditioned with masks

## Performance & Ablation Study
The author conducted detailed experiments with the following metrics as well as human evaluations.
- Shape score and texture score, each based on a Resnet-18 classifier of (shape or texture re- spectively);
- Shape AM score and texture AM score, based on the output of the same classifiers;
- Distance to nearest neighbors images from the training set;
- Texture and shape confusion of classifier;
- Darkness, average intensity and skewness of images;
