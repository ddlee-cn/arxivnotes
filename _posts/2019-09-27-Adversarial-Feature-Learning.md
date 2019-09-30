---
title: (BiGAN)Adversarial Feature Learning - Donahue - ICLR 2017
tag:
- GAN
- Image Generation
---

## Info

- Title: Adversarial Feature Learning
- Task: Image Generation
- Author: Jeff Donahue, Philipp Krähenbühl, Trevor Darrell
- Date: May 2016
- Arxiv: [1605.09782](https://arxiv.org/abs/1605.09782)
- Published: ICLR 2017



## Abstract

The ability of the Generative Adversarial Networks (GANs) framework to learn generative models mapping from simple latent distributions to arbitrarily complex data distributions has been demonstrated empirically, with compelling results showing that the latent space of such generators captures semantic variation in the data distribution. Intuitively, models trained to predict these semantic latent representations given data may serve as useful feature representations for auxiliary problems where semantics are relevant. However, in their existing form, GANs have no means of learning the inverse mapping -- projecting data back into the latent space. We propose Bidirectional Generative Adversarial Networks (BiGANs) as a means of learning this inverse mapping, and demonstrate that the resulting learned feature representation is useful for auxiliary supervised discrimination tasks, competitive with contemporary approaches to unsupervised and self-supervised feature learning.



## Motivation & Design

Similar to Adversarially learned inference (ALI), Bidirectional GAN (BiGAN) learn latent representations within the GAN framework combined with an encoder. They learn the joint probability distribution of data x and latent z while GAN learns only the data distribution directly. The discriminator receives samples from the joint space of the data x and the latent variable z and discriminates joint pairs (G(z), z) and (x, E(x)) where G and E represent a decoder and an encoder, respectively. By training an encoder and a decoder together, they can learn an inference X → Z while still being able to generate sharp, high-quality samples.

![BiGAN](https://i.imgur.com/sOa8L9l.png)




## Code

[Theano](<https://github.com/jeffdonahue/bigan>)



## Related

- [(ALI)Adversarially Learned Inference - ICLR 2017](https://arxivnote.ddlee.cn/2019/09/26/Adversarially-Learned-Inference-ICLR-2017.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/2019/08/18/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/2019/08/19/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs(from GAN to BigGAN)](https://arxivnote.ddlee.cn/2019/08/20/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)

