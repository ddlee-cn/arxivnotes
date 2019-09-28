---
title: "Xception: Deep Learning with Depthwise Seperable Convolutions - Chollet et al. - 2016"
tag:
- Image Classification
- CNN
redirect_from: /Xception-Deep-Learning-with-Depthwise-Seperable-Convolutions-Chollet-2016.html
---


## Info
- Title: **Xception: Deep Learning with Depthwise Seperable Convolutions**
- Author: F. Chollet
- Arxiv: [1610.02357](https://arxiv.org/abs/1610.02357)
- Date: Oct. 2016

## Highlights & Drawbacks
Replaced 1×1 convolution and 3×3 convolution in Inception unit with Depth-wise seperable convolution

<!-- more -->

## Motivation & Design

The article points out that the assumption behind the Inception unit is that the correlation between the channel and the space can be fully decoupled, similarly the convolution structure in the length and height directions (the 3 × 3 convolution in Inception-v3 is 1 × 3 and 3 × 1 convolution replacement).

Further, Xception is based on a stronger assumption: the correlation between channels and cross-space is completely decoupled. This is also the concept modeled by Depthwise Separable Convolution.
A simple Inception Module:

![Xception: Deep Learning with Depthwise Seperable Convolutions](https://i.imgur.com/voGGEeh.png)


is equal to:

![Xception: Deep Learning with Depthwise Seperable Convolutions](https://i.imgur.com/ttldnjQ.png)



Push # of channel to extreme, we obtain Depthwise Separable Convolution:

![Xception: Deep Learning with Depthwise Seperable Convolutions](https://i.imgur.com/2AuC4j9.png)


[NetScope Visualization](http://ethereon.github.io/netscope/#gist/931d7c91b22109f83bbbb7ff1a215f5f) and source code: [awesome_cnn](https://github.com/ddlee96/awesome_cnn).

![Xception: Deep Learning with Depthwise Seperable Convolutions](https://i.imgur.com/BT6sHIb.png)
