---
title: "(ResNeXt)Aggregated Residual Transformations for Deep Neural Networks - Xie et al. - CVPR 2017"
tag:
- CNN
- Image Classification
---

## Info
- Title: **Aggregated Residual Transformations for Deep Neural Networks**
- Task: **Image Classification**
- Author: S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. H
- Arxiv: [1611.05431](https://arxiv.org/abs/1611.05431)
- Date: Nov. 2016
- Published: CVPR 2017
- 1st Runner Up in ILSVRC 2016

## Highlights & Drawbacks
The core idea of ResNeXt is normalizing the multi-path structure of Inception module. Instead of using hand-designed 1x1, 3x3, and 5x5 convolutions, ResNeXt proposed a new hyper-parameter with reasonable meaning for network design.

The authors proposed a new dimension on designing neural network, which is called *cardinality*. Besides # of layers, # of channels, cardinality describes the count of paths inside one module. Compared to the Inception model, the paths share the exactly the same hyper-parameter. Additionally, short connection is added between layers.

<!-- more -->


## Motivation & Design

The three classical pattern on designing a neural network:

- **Repeat**: Starting with AlexNet and VGG, repeating the same structure is one of the most popular patterns of deep networks.
- **Multi-path**: Presented by the Inception-Series. Splitting inputs, transforming with multiple-size convolutions, then concatenation.
- **Skip-connection**: Applied to Image Recognition by ResNet. Simply rewriting the target function into identity mapping and residual function, allowing the interaction between shallow layers and deep layers.

The residual function is rewritten into:
$$
\mathbf{y}=\mathbf{x}+\sum_{i=1}^{C} \mathcal{T}_{i}(\mathbf{x}),
$$
C denotes the number of transformations(paths) inside the layer, a.k.a. cardinality.

![(ResNeXt)Aggregated Residual Transformations for Deep Neural Networks](https://i.imgur.com/JxJJiOH.png)


As the number of paths increases, the number of channel for each path is reduced to maintain capacity of network.

[NetScope Visualization](http://ethereon.github.io/netscope/#/gist/c2ba521fcb60520abb0b0da0e9c0f2ef) and source code(Pytorch+Caffe):[awesome_cnn](https://github.com/ddlee96/awesome_cnn).

## Performance & Ablation Study

![(ResNeXt)Aggregated Residual Transformations for Deep Neural Networks](https://i.imgur.com/I4Nhs1X.png)

![(ResNeXt)Aggregated Residual Transformations for Deep Neural Networks](https://i.imgur.com/FYvIp3v.png)


## Code
[Torch](https://github.com/facebookresearch/ResNeXt)