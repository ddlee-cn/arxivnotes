---
title: (Inception V3)Rethinking the Inception Architecture for Computer Vision - Szegedy et al. - 2015
tag:
- Image Classification
- CNN
---

# 

## Info
- Title: **Rethinking the Inception Architecture for Computer Vision**
- Author: C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna
- Arxiv: [1512.00567](https://arxiv.org/abs/1512.00567)
- Date: Dec. 2015

## Highlights & Drawbacks
This article is the author's step 2.5 to advance the inception structure. In an earlier article, the same author proposed Batch Normalization and used to improve the Inception structure, called Inception-BN. In this article, the authors proposed Inception-v2 and Inception-v3, both of which share the same network structure. The v3 version adds RMSProp, Label Smoothing and other techniques compared to the v2 version.

The article describes several design principles of the Inception series and improves the structure of GoogLeNet based on these principles.


<!-- more -->

## Motivation & Design

### General Design Principles

- Avoid representational bottlenecks, especially early in the network. It is not recommended to perform feature compression at too shallow a stage, and dimension is only a reference for expressing complexity and cannot be used as an absolute measure of feature complexity.
- Higher dimensional representations are easier to process locally with a network. Higher order representations have more local descriptive forces, and increasing nonlinearity helps to solidify these descriptive forces.
- Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power. Space-based aggregated information can be processed in a low-dimensional space without fear of too much information loss. This also supports the dimensionality reduction of the 1×1 convolution.
- The increase in width and depth contributes to the expressive power of the network. The best practice is to advance in both directions, rather than just one.

### Factorizing Convolution
Decomposition has always been a classic idea in computational mathematics. From Newton's method to BFGS, the Hessian matrix (or its inverse) is represented and approximated by a series of vector operations to avoid matrix calculations.

This paper proposes a decomposition of two convolutional structures, one at the level of the convolution kernel and the other in terms of space.

The first decomposition is to solve the large nuclear volume integration into a series of small nuclear convolutions.

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/IIUYBbQ.png)


Replace the 5 × 5 convolution with two 3 × 3 convolutions, and the parameter reduction is (9 + 9) / (5 × 5).

The second decomposition is to introduce an asymmetric convolution on the convolution kernel itself: replace the 3x3 convolution with a 3x1 and 1x3 convolution series. As shown below.

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/Z304W8I.png)


This decomposition can also be generalized to the n-dimensional case, and the larger the n, the more obvious the benefits.

The volume integral solution on the space models a situation where the convolution parameters in both directions are orthogonal to each other and are decoupled by spatial decomposition convolution.

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/HDS4SsX.png)


### Utility of Auxiliary Classifiers

In GoogLeNet, the author supervised the learning of low-dimensional feature maps with loss, but further experiments found that after adding the BN layer, these gains were offset, so the Auxiliary Classifier can be seen as some sort of regularization technique. It is no longer used in the BN network.

### Efficient Grid Size Reduction

This section discusses the feature dimension reduction in the network, the process of downsampling, usually controlled by the stride parameter of the convolutional layer or the Pooling layer. In order to avoid the Representation Bottleneck mentioned in Principle 1, the network is widened (through the increase in the number of channels) before the Pooling, which also corresponds to the principle of balancing the width and depth.

The elements that ultimately combine the Inception structure and the downsampling requirements are as follows:

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/LdSwET7.png)

Unlike the Inception unit, the above 1×1 convolution extends Channel, and the 3×3 convolution uses stride=2.
### Inception-v2 & Inception-v3 Architecture

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/cwamcBa.png)



It can be seen that as the depth increases, the number of Channels also expands, and the Inception unit also follows the stacking paradigm.

Three of the Inception units are:

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/odzHOvd.png)

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/LaK7gzh.png)

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/MoqrFPT.png)



Alternatively, you can check out the structure of Inception-v3 by looking at [NetScope Vis] (http://ethereon.github.io/netscope/#gist/a2394c1c4a9738469078f096a8979346), the source file is located at [awesome_cnn] (https://github.com/ Ddle96/awesome_cnn).

## Performance & Ablation Study

The following is the gain decomposition brought about by the evolution of the Inception structure:

![ (Inception V3)Rethinking the Inception Architecture for Computer Vision](https://i.imgur.com/bqCZESD.png)


## Misc
For a further understanding of convolution, refer to this [page] (https://graphics.stanford.edu/courses/cs178-10/applets/convolution.html), which visualizes the processing of different convolution check inputs. The examples given are those that are manually designed in the early days, and the deep network implicitly learned the convolutional expression of these filters.

