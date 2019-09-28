---
title: "Glow: Generative Flow with Invertible 1x1 Convolutions - Kingma & Dhariwal - NIPS 2018"
tag:
- AutoRegressive
- Image Generation
---



## Info
- Title: **Glow: Generative Flow with Invertible 1x1 Convolutions**
- Task: **Image Generation**
- Author: D. P. Kingma and P. Dhariwal
- Date: Jul. 2018
- Arxiv: [1807.03039](https://arxiv.org/abs/1807.03039)
- Published: NIPS 2018

## Motivation & Design
###The merits of flow-based models
- Exact latent-variable inference and log-likelihood evaluation. In VAEs, one is able to infer only approximately the value of the latent variables that correspond to a datapoint. GAN’s have no encoder at all to infer the latents. In reversible generative models, this can be done exactly without approximation. Not only does this lead to accurate inference, it also enables optimization of the exact log-likelihood of the data, instead of a lower bound of it.
- Efficient inference and efficient synthesis. Autoregressive models, such as the Pixel- CNN (van den Oord et al., 2016b), are also reversible, however synthesis from such models is difficult to parallelize, and typically inefficient on parallel hardware. Flow-based generative models like Glow (and RealNVP) are efficient to parallelize for both inference and synthesis.
- Useful latent space for downstream tasks. The hidden layers of autoregressive models have unknown marginal distributions, making it much more difficult to perform valid manipulation of data. In GANs, data points can usually not be directly represented in a latent space, as they have no encoder and might not have full support over the data distribution. (Grover et al., 2018). This is not the case for reversible generative models and VAEs, which allow for various applications such as interpolations between data points and meaningful modifications of existing data points.
- Significant potential for memory savings. Computing gradients in reversible neural networks requires an amount of memory that is constant instead of linear in their depth.

### The proposed flow
![CleanShot 2019-08-20 at 15.25.05@2x](https://i.imgur.com/L797JWn.jpg)

The authors propose a generative flow where each step (left) consists of an actnorm step, followed by an invertible 1 × 1 convolution, followed by an affine transformation (Dinh et al., 2014). This flow is combined with a multi-scale architecture (right).

There are three steps in one stage of flow in Glow.

Step 1:**Activation normalization** (short for “actnorm”)

It performs an affine transformation using a scale and bias parameter per channel, similar to batch normalization, but works for mini-batch size 1. The parameters are trainable but initialized so that the first minibatch of data have mean 0 and standard deviation 1 after actnorm.

Step 2: **Invertible 1x1 conv**

Between layers of the RealNVP flow, the ordering of channels is reversed so that all the data dimensions have a chance to be altered. A 1×1 convolution with equal number of input and output channels is a generalization of any permutation of the channel ordering.

Say, we have an invertible 1x1 convolution of an input $h×w×c$ tensor $h$ with a weight matrix $W$ of size $c×c$. The output is a $h×w×c$ tensor, labeled as $ f=𝚌𝚘𝚗𝚟𝟸𝚍(h;W)$. In order to apply the change of variable rule, we need to compute the Jacobian determinant $|det∂f/∂h|$.

Both the input and output of 1x1 convolution here can be viewed as a matrix of size $h×w$. Each entry $x_{ij}$($i=1,2...h, j=1,2,...,w$) in $h$ is a vector of $c$ channels and each entry is multiplied by the weight matrix $W$ to obtain the corresponding entry $y_{ij}$ in the output matrix respectively. The derivative of each entry is $\partial \mathbf{x}_{i j} \mathbf{W} / \partial \mathbf{x}_{i j}=\mathbf{w}$ and there are $h×w$ such entries in total:

The inverse 1x1 convolution depends on the inverse matrix $W^{−1}$
. Since the weight matrix is relatively small, the amount of computation for the matrix determinant (tf.linalg.det) and inversion (tf.linalg.inv) is still under control.

Step 3: Affine coupling layer

The design is same as in RealNVP.

![CleanShot 2019-08-20 at 11.17.30@2x](https://i.imgur.com/kqWmlSn.jpg)

The three main components of proposed flow, their reverses, and their log-determinants. Here, $x$ signifies the input of the layer, and $y$ signifies its output. Both $x$ and $y$ are tensors of shape $[h × w × c]$ with spatial dimensions (h, w) and channel dimension $c$. With $(i, j)$ we denote spatial indices into tensors $x$ and $y$. The function NN() is a nonlinear mapping, such as a (shallow) convolutional neural network like in ResNets (He et al., 2016) and RealNVP (Dinh et al., 2016).


## Performance & Ablation Study
![CleanShot 2019-08-20 at 11.18.32@2x](https://i.imgur.com/NEX73i5.jpg)

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


## Code
- [Blog Post](https://openai.com/blog/glow)
- [TensorFlow](https://github.com/openai/glow)

## Related
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- Gated PixelCNN: [Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016](https://arxivnote.ddlee.cn/Conditional-Image-Generation-with-PixelCNN-Decoders.html)
- PixelRNN & PixelCNN: [Pixel Recurrent Neural Networks - van den Oord - ICML 2016](https://arxivnote.ddlee.cn/Pixel-Recurrent-Neural-Networks.html)
- VQ-VAE: [Neural Discrete Representation Learning - van den Oord - NIPS 2017](https://arxivnote.ddlee.cn/Neural-Discrete-Representation-Learning.html)
- VQ-VAE-2: [ Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019](https://arxivnote.ddlee.cn/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2.html)