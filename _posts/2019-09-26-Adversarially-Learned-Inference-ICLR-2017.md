---
title: (ALI)Adversarially Learned Inference - Dumoulin - ICLR 2017
tag:
- GAN
- Image Generation
---

## Info

- Title: Adversarially Learned Inference
- Task: Image Generation
- Author: Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Alex Lamb, Martin Arjovsky, Olivier Mastropietro and Aaron Courville
- Date: June 2016
- Arxiv: [1606.00704](https://arxiv.org/abs/1606.00704)
- Published: ICLR 2017



## Abstract

We introduce the adversarially learned inference (ALI) model, which jointly learns a generation network and an inference network using an adversarial process. The generation network maps samples from stochastic latent variables to the data space while the inference network maps training examples in data space to the space of latent variables. An adversarial game is cast between these two networks and a discriminative network is trained to distinguish between joint latent/data-space samples from the generative network and joint samples from the inference network. We illustrate the ability of the model to learn mutually coherent inference and generation networks through the inspections of model samples and reconstructions and confirm the usefulness of the learned representations by obtaining a performance competitive with state-of-the-art on the semi-supervised SVHN and CIFAR10 tasks.



## Motivation & Design

The adversarially learned inference (ALI) model is a deep directed generative model which jointly learns a generation network and an inference network using an adversarial process. This model constitutes a novel approach to integrating efficient inference with the generative adversarial networks (GAN) framework.

What makes ALI unique is that unlike other approaches to learning inference in deep directed generative models (like variational autoencoders (VAEs)), the objective function involves no explicit reconstruction loop. Instead of focusing on achieving a pixel-perfect reconstruction, ALI tends to produce believable reconstructions with interesting variations, albeit at the expense of making some mistakes in capturing exact object placement, color, style and (in extreme cases) object identity. This is a good thing, because 1) capacity is not wasted to model trivial factors of variation in the input, and 2) the learned features are more or less invariant to these trivial factors of variation, which is what is expected of good feature learning.

These strenghts are showcased via the semi-supervised learning tasks on SVHN and CIFAR10, where ALI achieves a performance competitive with state-of-the-art.


Even though GANs are pretty good at producing realistic-looking synthetic samples, they lack something very important: the ability to do inference.

Inference can loosely be defined as the answer to the following question:

Given x, what z is likely to have produced it?

This question is exactly what ALI is equipped to answer.

ALI augments GAN’s generator with an additional network. This network receives a data sample as input and produces a synthetic z as output.

Expressed in probabilistic terms, ALI defines two joint distributions:

- the encoder joint $q(\mathbf{x}, \mathbf{z}) = q(\mathbf{x})q(\mathbf{z} \mid \mathbf{x})$ and
- the decoder joint $p(\mathbf{x}, \mathbf{z}) = p(\mathbf{z})p(\mathbf{x} \mid \mathbf{z})$.

ALI also modifies the discriminator’s goal. Rather than examining x samples marginally, it now receives joint pairs $(x, z)$ as input and must predict whether they come from the encoder joint or the decoder joint.

Like before, the generator is trained to fool the discriminator, but this time it can also learn $q(z∣x)$.

![ALI, Adversariallly Learned Inference](https://ishmaelbelghazi.github.io/ALI/assets/ali_probabilistic.svg)


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



The adversarial game played between the discriminator and the generator is formalized by the following value function:

$$
% <![CDATA[
\begin{split}
    \min_G \max_D V(D, G)
    &= \mathbb{E}_{q(\mathbf{x})} [\log(D(\mathbf{x}, G_z(\mathbf{x})))] +
       \mathbb{E}_{p(\mathbf{z})} [\log(1 - D(G_x(\mathbf{z}), \mathbf{z}))] \\
    &= \iint q(\mathbf{x}) q(\mathbf{z} \mid \mathbf{x})
             \log(D(\mathbf{x}, \mathbf{z})) d\mathbf{x} d\mathbf{z} \\
    &+ \iint p(\mathbf{z}) p(\mathbf{x} \mid \mathbf{z})
             \log(1 - D(\mathbf{x}, \mathbf{z})) d\mathbf{x} d\mathbf{z}
\end{split} %]]>
$$

In analogy to GAN, it can be shown that for a fixed generator, the optimal discriminator is
$$
D^*(\mathbf{x}, \mathbf{z}) = \frac{q(\mathbf{x}, \mathbf{z})}
                                       {q(\mathbf{x}, \mathbf{z}) +
                                        p(\mathbf{x}, \mathbf{z})}
$$

and that given an optimal discriminator, minimizing the value function with respect to the generator parameters is equivalent to minimizing the Jensen-Shannon divergence between $p(x,z)$ and $q(x,z)$.

Matching the joints also has the effect of matching the marginals (i.e., $p(x) \ sim q(x)$ and $p(z) \sim q(z)$) as well as the conditionals /posteriors (i.e., $p(z|x) \sim q(z|x)$ and $q(x|z) \sim p(x|z)$).



## Experiments & Ablation Study



## CIFAR10

The [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset contains 60,000 32x32 colour images in 10 classes.

| ![img](https://ishmaelbelghazi.github.io/ALI/assets/cifar10_samples.png) | ![img](https://ishmaelbelghazi.github.io/ALI/assets/cifar10_reconstructions.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Samples                                                      | Reconstructions                                              |



## SVHN

[SVHN](http://ufldl.stanford.edu/housenumbers/) is a dataset of digit images obtained from house numbers in Google Street View images. It contains over 600,000 labeled examples.

| ![img](https://ishmaelbelghazi.github.io/ALI/assets/svhn_samples.png) | ![img](https://ishmaelbelghazi.github.io/ALI/assets/svhn_reconstructions.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Samples                                                      | Reconstructions                                              |



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


## CelebA

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a dataset of celebrity faces with 40 attribute annotations. It contains over 200,000 labeled examples.

| ![img](https://ishmaelbelghazi.github.io/ALI/assets/celeba_samples.png) | ![img](https://ishmaelbelghazi.github.io/ALI/assets/celeba_reconstructions.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Samples                                                      | Reconstructions                                              |



## Tiny ImageNet

The Tiny Imagenet dataset is a version of the [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/) dataset that has been center-cropped and downsampled to 64×6464×64 pixels. It contains over 1,200,000 labeled examples.

| ![img](https://ishmaelbelghazi.github.io/ALI/assets/tiny_imagenet_samples.png) | ![img](https://ishmaelbelghazi.github.io/ALI/assets/tiny_imagenet_reconstructions.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Samples                                                      | Reconstructions                                              |




## Code

[Project Site](https://ishmaelbelghazi.github.io/ALI/)

[Theano](https://github.com/IshmaelBelghazi/ALI)

[OpenReview](https://openreview.net/forum?id=B1ElR4cggS)

## Related

- [(BiGAN)Adversarial Feature Learning - ICLR 2017](https://arxivnote.ddlee.cn/2019/09/27/Adversarial-Feature-Learning.html)

- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/2019/08/18/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/2019/08/19/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs(from GAN to BigGAN)](https://arxivnote.ddlee.cn/2019/08/20/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)

