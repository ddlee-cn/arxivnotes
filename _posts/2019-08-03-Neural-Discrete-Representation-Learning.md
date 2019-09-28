---
title: (VQ-VAE)Neural Discrete Representation Learning - van den Oord - NIPS 2017 - TensorFlow & PyTorch Code
tag:
- VAE
- Image Generation
---



## Info
- Title: **Neural Discrete Representation Learning**
- Task: **Image Generation**
- Author: A. van den Oord, O. Vinyals, and K. Kavukcuoglu
- Date: Nov. 2017
- Arxiv: [1711.00937](https://arxiv.org/abs/1711.00937)
- Published: NIPS 2017
- Affiliation: Google DeepMind


## Highlights & Drawbacks
- Discrete representation for data distribution
- The prior is learned instead of random

## Motivation & Design

### Vector Quantisation(VQ)
Vector quantisation (VQ) is a method to map $K$-dimensional vectors into a finite set of “code” vectors. The encoder output $E(\mathbf{x})=\mathbf{z}_{e}$ goes through a nearest-neighbor lookup to match to one of $K$ embedding vectors and then this matched code vector becomes the input for the decoder $D(.)$:

$$
z_{q}(x)=e_{k}, \quad \text { where } \quad k=\operatorname{argmin}_{j}\left\|z_{e}(x)-e_{j}\right\|_{2}
$$

The dictionary items are updated using Exponential Moving Averages(EMA), which is similar to EM methods like K-Means.

![(VQ-VAE)Neural Discrete Representation Learning](https://i.imgur.com/O8c2e05.png)


### Loss Design
- Reconstruction loss
- VQ loss: The L2 error between the embedding space and the encoder outputs.
- Commitment loss: A measure to encourage the encoder output to stay close to the embedding space and to prevent it from fluctuating too frequently from one code vector to another.

$$
L=\underbrace{\left\|\mathbf{x}-D\left(\mathbf{e}_{k}\right)\right\|_{2}^{2}}_{\text { reconstruction loss }}+\underbrace{\left\|\operatorname{sg}[E(\mathbf{x})]-\mathbf{e}_{k}\right\|_{2}^{2}}_{\text { VQ loss }}+\underbrace{\beta\left\|E(\mathbf{x})-\operatorname{sg}\left[\mathbf{e}_{k}\right]\right\|_{2}^{2}}_{\text { commitment loss }}
$$

where sq[.] is the  `stop_gradient`  operator.

### Prior
Training PixelCNN and WaveNet for images and audio respectively on learned latent space, the VA-VAE model avoids "posterior collapse" problem which VAE suffers from.


## Performance & Ablation Study

Original VS. Reconstruction:
![(VQ-VAE)Neural Discrete Representation Learning](https://i.imgur.com/oUgQSUz.png)


Category Image Generation:
![(VQ-VAE)Neural Discrete Representation Learning](https://i.imgur.com/k3CzNQl.png)



## Code
- [PyTorch](https://github.com/zalandoresearch/pytorch-vq-vae)
- [Sonnet Implementation](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py)
- [Sonnet Example](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb)
- [Project Site](https://avdnoord.github.io/homepage/vqvae/)

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
- VQ-VAE-2: [ Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019](https://arxivnote.ddlee.cn/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2.html)
- [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification - Salimans - ICLR 2017](https://arxivnote.ddlee.cn/PixelCNN++-Improving-the-PixelCNN-with-Discretized-Logistic-Mixture-Likelihood-and-Other-Modification.html)
- Gated PixelCNN: [Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016](https://arxivnote.ddlee.cn/Conditional-Image-Generation-with-PixelCNN-Decoders.html)
- PixelRNN & PixelCNN: [Pixel Recurrent Neural Networks - van den Oord - ICML 2016](https://arxivnote.ddlee.cn/Pixel-Recurrent-Neural-Networks.html)