---
title: A Neural Algorithm of Artistic Style - Gatys et al. - CVPR 2016
tag:
- Style Transfer
- Art

---


## Info
- Title: **A Neural Algorithm of Artistic Style**
- Task: **Style Transfer**
- Author: L. A. Gatys, A. S. Ecker, and M. Bethge
- Arxiv: [1508.06576](http://arxiv.org/abs/1508.06576)
- Date: Aug. 2015
- Published: CVPR 2016

## Motivation & Design

To obtain a representation of the style of an input image, we use a feature space originally designed to capture texture information.

**Key Finding**
the representations of content and style in the Convolutional Neural Network are separable.

<!-- more -->

### Network Architecture

![A Neural Algorithm of Artistic Style](https://i.imgur.com/uhDNbZk.png)

### Content Match
To visualise the image information that is encoded at different layers of the hierarchy (Fig 1, content reconstructions) we perform gradient descent on a white noise image to find another image that matches the feature responses of the original image.

$$
\mathcal{L}\_{\text {content}}(\vec{p}, \vec{x}, l)=\frac{1}{2} \sum_{i, j}\left(F_{i j}^{l}-P_{i j}^{l}\right)^{2}
$$

### Style Match
  These feature correlations are given by the Gram matrix $G^l \in R\_N^l×N^l$, where $G^l\_{ij}$ is the inner product between the vectorised feature map of $i$ and $ j $ in layer $ l $:


$$
G_{i j}^{l}=\sum_{k} F_{i k}^{l} F_{j k}^{l}
$$
To generate a texture that matches the style of a given image, we use gradient descent from a white noise image to find another image that matches the style representation of the original image. This is done by minimising the mean-squared distance between the entries of the Gram matrix from the original image and the Gram matrix of the image to be generated.

$$
E_{l}=\frac{1}{4 N_{l}^{2} M_{l}^{2}} \sum_{i, j}\left(G_{i j}^{l}-A_{i j}^{l}\right)^{2}
$$

$$
\mathcal{L}\_{s t y l e}(\vec{a}, \vec{x})=\sum_{l=0}^{L} w_{l} E_{l}
$$


### Total Loss

$$
\mathcal{L}\_{\text {total}}(\vec{p}, \vec{a}, \vec{x})=\alpha \mathcal{L}\_{\text {content}}(\vec{p}, \vec{x})+\beta \mathcal{L}_{\text {style}}(\vec{a}, \vec{x})
$$

## Code
- [TensorFlow](https://github.com/anishathalye/neural-style)
- [PyTorch](https://github.com/pytorch/examples/tree/master/fast_neural_style)



