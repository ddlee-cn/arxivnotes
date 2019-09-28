---
title: "Deep Generative Models(Part 1): Taxonomy and VAEs"
tag:
- Review
- VAE
- Image Generation
---



A Generative Model learns a probability distribution from data with prior knowledge, producing new images from learned distribution.

![Deep Generative Models: A Taxonomy](https://i.imgur.com/yQfmFIT.png)



[TOC]



## Key choices
### Representation
There are two main choices for learned representation: **factorized model** and **latent variable model**.

**Factorized model** writes probability distribution as a product of simpler terms, via chain rule.
![Deep Generative Models: A Taxonomy](https://i.imgur.com/5n5o8yl.png)

**Latent variable model** defines a latent space to extract the core information from data, which is much smaller than the original one.

![Deep Generative Models: A Taxonomy](https://i.imgur.com/U4o2qem.png)

### Learning
**Max Likelihood Estimation**
- fully-observed graphical models: PixelRNN & PixelCNN -> PixelCNN++, WaveNet(audio)
- latent-variable models: VAE -> VQ-VAE
- latent-variable invertible models(Flow-based): NICE, Real NVP -> MAF, IAF, Glow

**Adversarial Training**
- GANs: Vanilla GAN -> improved GAN, DCGAN, cGAN -> WGAN, ProGAN -> SAGAN, StyleGAN, BigGAN


Comparison of GAN, VAE and Flow-based Models
![Deep Generative Models: A Taxonomy](https://i.imgur.com/brtrqi4.png)

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



## VAE: Variational AutoEncoder

### Auto-Encoding Variational Bayes - Kingma - ICLR 2014
- Title: **Auto-Encoding Variational Bayes**
- Task: **Image Generation**
- Author: D. P. Kingma and M. Welling
- Date: Dec. 2013
- Arxiv: [1312.6114](https://arxiv.org/abs/1312.6114)
- Published: ICLR 2014

**Highlights**
- A reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods
- For i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator

The key idea: approximate the posterior $p_θ(z|x)$ with a simpler, tractable distribution $q_ϕ(z|x)$.
![Auto-Encoding Variational Bayes - Kingma - ICLR 2014](https://i.imgur.com/LE4oMbx.jpg)

The graphical model involved in Variational Autoencoder. Solid lines denote the generative distribution $p_θ(.)$ and dashed lines denote the distribution $q_ϕ(z|x)$ to approximate the intractable posterior $p_θ(z|x)$.

![Auto-Encoding Variational Bayes - Kingma - ICLR 2014](https://i.imgur.com/26oV5mL.jpg)

**Loss Function: ELBO**
Using KL Divergence:
$$
D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z} | \mathbf{x})\right)=\log p_{\theta}(\mathbf{x})+D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z})\right)-\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z})
$$

ELOB defined as:
$$
\begin{aligned} L_{\mathrm{VAE}}(\theta, \phi) &=-\log p_{\theta}(\mathbf{x})+D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z} | \mathbf{x})\right) \\ &=-\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} | \mathbf{x})} \log p_{\theta}(\mathbf{x} | \mathbf{z})+D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z} | \mathbf{x}) \| p_{\theta}(\mathbf{z})\right) \\ \theta^{*}, \phi^{*} &=\arg \min _{\theta, \phi} L_{\mathrm{VAE}} \end{aligned}
$$

By minimizing the loss we are maximizing the lower bound of the probability of generating real data samples.

**The Reparameterization Trick**

The expectation term in the loss function invokes generating samples from $z∼q_ϕ(z|x)$. Sampling is a stochastic process and therefore we cannot backpropagate the gradient. To make it trainable, the reparameterization trick is introduced: It is often possible to express the random variable $z$
 as a deterministic variable $\mathbf{z}=\mathcal{T}_{\phi}(\mathbf{x}, \boldsymbol{\epsilon})$, where $ϵ$ is an auxiliary independent random variable, and the transformation function $\mathcal{T}_{\phi}$ parameterized by $ϕ$ converts $ϵ$ to $z$.

For example, a common choice of the form of $q_ϕ(z|x)$ ltivariate Gaussian with a diagonal covariance structure:
$$
\begin{array}{l}{\mathbf{z} \sim q_{\phi}\left(\mathbf{z} | \mathbf{x}^{(i)}\right)=\mathcal{N}\left(\mathbf{z} ; \boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)} \boldsymbol{I}\right)} \\ {\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \text { where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})}\end{array}
$$
where $⊙$ refers to element-wise product.

![Auto-Encoding Variational Bayes - Kingma - ICLR 2014](https://i.imgur.com/61JQRa8.jpg)

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

### (VQ-VAE)Neural Discrete Representation Learning - van den Oord - NIPS 2017 
- Title: **Neural Discrete Representation Learning**
- Task: **Image Generation**
- Author: A. van den Oord, O. Vinyals, and K. Kavukcuoglu
- Date: Nov. 2017
- Arxiv: [1711.00937](https://arxiv.org/abs/1711.00937)
- Published: NIPS 2017
- Affiliation: Google DeepMind


**Highlights**
- Discrete representation for data distribution
- The prior is learned instead of random


**Vector Quantisation(VQ)**
Vector quantisation (VQ) is a method to map $K$-dimensional vectors into a finite set of “code” vectors. The encoder output $E(\mathbf{x})=\mathbf{z}_{e}$ goes through a nearest-neighbor lookup to match to one of $K$ embedding vectors and then this matched code vector becomes the input for the decoder $D(.)$:

$$
z_{q}(x)=e_{k}, \quad \text { where } \quad k=\operatorname{argmin}_{j}\left\|z_{e}(x)-e_{j}\right\|_{2}
$$

The dictionary items are updated using Exponential Moving Averages(EMA), which is similar to EM methods like K-Means.

![(VQ-VAE)Neural Discrete Representation Learning](https://i.imgur.com/O8c2e05.png)


**Loss Design**
- Reconstruction loss
- VQ loss: The L2 error between the embedding space and the encoder outputs.
- Commitment loss: A measure to encourage the encoder output to stay close to the embedding space and to prevent it from fluctuating too frequently from one code vector to another.

$$
L=\underbrace{\left\|\mathbf{x}-D\left(\mathbf{e}_{k}\right)\right\|_{2}^{2}}_{\text { reconstruction loss }}+\underbrace{\left\|\operatorname{sg}[E(\mathbf{x})]-\mathbf{e}_{k}\right\|_{2}^{2}}_{\text { VQ loss }}+\underbrace{\beta\left\|E(\mathbf{x})-\operatorname{sg}\left[\mathbf{e}_{k}\right]\right\|_{2}^{2}}_{\text { commitment loss }}
$$

where sq[.] is the  `stop_gradient`  operator.

Training PixelCNN and WaveNet for images and audio respectively on learned latent space, the VA-VAE model avoids "posterior collapse" problem which VAE suffers from.

### Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019

- Title: **Generating Diverse High-Fidelity Images with VQ-VAE-2**
- Task: **Image Generation**
- Author: A. Razavi, A. van den Oord, and O. Vinyals
- Date: Jun. 2019
- Arxiv: [1906.00446](https://arxiv.org/abs/1906.00446)
- Affiliation: Google DeepMind

**Highlights**
- Diverse generated results
- A multi-scale hierarchical organization of VQ-VAE
- Self-attention mechanism over autoregressive model


![Generating Diverse High-Fidelity Images with VQ-VAE-2](https://i.imgur.com/kNEGBCj.png)

**Stage 1: Training hierarchical VQ-VAE**
The design of hierarchical latent variables intends to separate local patterns (i.e., texture) from global information (i.e., object shapes). The training of the larger bottom level codebook is conditioned on the smaller top level code too, so that it does not have to learn everything from scratch.

![Generating Diverse High-Fidelity Images with VQ-VAE-2](https://i.imgur.com/HmBVGcm.png)


**Stage 2: Learning a prior over the latent discrete codebook**
The decoder can receive input vectors sampled from a similar distribution as the one in training. A powerful autoregressive model enhanced with multi-headed self-attention layers is used to capture the correlations in spatial locations that are far apart in the image with a larger receptive field. 

![Generating Diverse High-Fidelity Images with VQ-VAE-2](https://i.imgur.com/kbiYRcN.png)

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


## References
1. [IJCAI 2018 Tutorial: Deep Generative Models](https://ermongroup.github.io/generative-models/)
2. [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)

## Related
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- Gated PixelCNN: [Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016](https://arxivnote.ddlee.cn/Conditional-Image-Generation-with-PixelCNN-Decoders.html)
- PixelRNN & PixelCNN: [Pixel Recurrent Neural Networks - van den Oord - ICML 2016](https://arxivnote.ddlee.cn/Pixel-Recurrent-Neural-Networks.html)
- [Glow: Generative Flow with Invertible 1x1 Convolutions - Kingma & Dhariwal - NIPS 2018](https://arxivnote.ddlee.cn/Glow-Generative-Flow-with-Invertible-1x1-Convolutions-Kingma-Dhariwal-NIPS.html)
- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)