---
title: (DMIT)Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019
tag:
- GAN
- Image-to-Image Translation

---

## Info

- Title: Multi-mapping Image-to-Image Translation via Learning Disentanglement
- Task: Image Translation
- Author:
- Date: Sep. 2019
- Arxiv: [1909.07877](https://arxiv.org/abs/1909.07877)
- Published: NIPS 2019



## Highlights

Multi-modal multi-domain translation with unified structure.

## Abstract

Recent advances of image-to-image translation focus on learning the one-to-many mapping from two aspects: multi-modal translation and multi-domain translation. Hothe authorsver, the existing methods only consider one of the two perspectives, which makes them unable to solve each other's problem. To address this issue, the authors propose a novel unified model, which bridges these two objectives. First, the authors disentangle the input images into the latent representations by an encoder-decoder architecture with a conditional adversarial training in the feature space. Then, the authors encourage the generator to learn multi-mappings by a random cross-domain translation. As a result, the authors can manipulate different parts of the latent representations to perform multi-modal and multi-domain translations simultaneously. Experiments demonstrate that our method outperforms state-of-the-art methods.



## Motivation & Design

**Comparisons of unsupervised I2I translation methods**

Denote $X_k$ as the k-th image domain. The solid lines and dashed lines represent the flow of encoder and generator respectively. The lines with the same color indicate they belong to the same module.



![Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://i.imgur.com/eRtVG3C.png)




### Method

As illustrated below, the authors introduce the content encoder $E_c : X → C$ that maps an input image to
its content, and the encoder style $E_s : X → S$ that extracts the style of the input image. To unify the
formulation, the authors also denote the determined mapping function betthe authorsen X and D as the domain label
encoder $E_d : X → D$ which is organized as a dictionary and extracts the domain label from the
input image. The inversely disentangled mapping is formulated as the generator $G : (C, S, D) → X$ .
As a result, with any desired style $s ∈ S$ and domain label $d ∈ D$, the authors can translate an input image
$x_i ∈ X$ to the corresponding target $x_t ∈ X$


$$
x_{t}=G\left(E_{c}\left(x_{i}\right), s, d\right)
$$

![Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://i.imgur.com/X1ie2m6.jpg)

#### Disentanglement path

To align the style representations across visual domains and constrain the information of the styles, the authors encourage the distribution of styles of all domains to be as close as possible to a prior distribution.


$$
\mathcal{L}_{c V A E}=\lambda_{K L} \mathbb{E}_{x_{i} \sim \mathcal{X}}\left[K L\left(E_{s}\left(x_{i}\right) \| q(s)\right]+\lambda_{r e c} \mathbb{E}_{x_{i} \sim \mathcal{X}}\left[\left\|G\left(E_{c}\left(x_{i}\right), E_{s}\left(x_{i}\right), E_{d}\left(x_{i}\right)\right)-x_{i}\right\|_{1}\right]\right.
$$



![Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://i.imgur.com/djvY262.jpg)




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


To enable stochastic sampling at test time, the authors choose the prior distribution q(s) to be a standard
Gaussian distribution $N (0, I)$. As for the content representations, the authors propose to perform conditional adversarial training in the content space to address the distribution shift issue of the contents among domains. This process encourages Ec to exclude the information of the domain d in content c


$$
\mathcal{L}_{G A N}^{c}=\mathbb{E}_{x_{i} \sim \mathcal{X}}\left[\log \left(D_{c}\left(E_{c}\left(x_{i}\right), E_{d}\left(x_{i}\right)\right)\right)+\mathbb{E}_{\left.d \sim\left(\mathcal{A}_{i}\right)\right\}}\left[\log \left(1-D_{c}\left(E_{c}\left(x_{i}\right), d\right)\right)\right]\right]
$$



![Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://i.imgur.com/WY7DO9Y.jpg)



#### Translation path:

The authors first propose to randomly sample domain labels and styles in the prior distributions, in order to cover the whole sampling space at training time. Then the authors introduce the latent regression [2, 45] to force the generator to utilize the style vector. The regression can also be applied to the content c to separate the style s from c. Thus the latent regression can be written as


$$
\mathcal{L}_{r e g}=\mathbb{E}_{s \sim \mathcal{N} \atop d \sim \mathcal{D}}^{c \sim c}\left[\left\|E_{s}(G(c, s, d))-c
s\right\|_{1}\right]+\mathbb{E}_{s \sim \mathcal{N} \atop d \sim \mathcal{D}}^{c \sim c}\left[\left\|E_{c}(G(c, s, d))-c\right\|_{1}\right]
$$



![Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://i.imgur.com/eiVLzkB.jpg)



To match the distribution of generated images to the real data with sampling domain labels and styles, the authors employ conditional adversarial training in the pixel space


$$
\begin{aligned} \mathcal{L}_{G A N}^{x}=& \mathbb{E}_{x_{i} \sim \mathcal{X}}\left[\log \left(D_{x}\left(x_{i}, E_{d}\left(x_{i}\right)\right)\right)+\mathbb{E}_{d \sim\left(\mathcal{D}-\left\{E_{d}\left(x_{i}\right)\right\}\right)}\left[\frac{1}{2} \log \left(1-D_{x}\left(x_{i}, d\right)\right)\right.\right.\\ &\left.+\mathbb{E}_{s \sim \mathcal{N}}\left[\frac{1}{2} \log \left(1-D_{x}\left(G\left(E_{c}\left(x_{i}\right), s, d\right)\right)\right]\right]\right] \end{aligned}
$$



![Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://i.imgur.com/gIRMP4p.jpg)



By combining both training paths, the full objective function of  model is


$$
\min _{G, E_{c}, E_{s}} \max _{D_{c}, D_{x}} \mathcal{L}_{D-P a t h}+\mathcal{L}_{T-P a t h} \\
\mathcal{L}_{T-P a t h}=\lambda_{r e g} \mathcal{L}_{r e g}+\mathcal{L}_{G A N}^{x} \\
\mathcal{L}_{D-P a t h}=\mathcal{L}_{c V A E}+\mathcal{L}_{G A N}^{c}
$$





## Experiments & Ablation Study

### Season Transfer

![Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://i.imgur.com/n7jhbrG.png)

To quantify the performance, the authors first translate each test image to 10 targets by sampling styles from prior distributions. Then the authors adopt Fréchet Inception Distance (FID) to evaluate the quality of generated images, and LPIPS (official version 0.1) to measure the diversity of samples generated by same input image.

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





### Semantic image synthesis

![Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://i.imgur.com/7vwtXUc.png)



In each column, the first row is the input image and the remaining rows are the outputs according to the above text description. In each pair of images generated by DMIT, the images in the first column are generated by encoding the style from the input image and the second column are generated by random style.



## Code

[Project Site](https://github.com/Xiaoming-Yu/DMIT)



## Related

- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)

