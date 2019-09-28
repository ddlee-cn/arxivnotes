---
title: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning - Szegedy et al. - 2016
tag:
- CNN
- Image Classification
---

## Info
- Title: **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning**
- Task: **Image Classification**
- Author: C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi
- Arxiv: [1602.07261](https:/arXiv.org/abs/1602.0726)
- Date: Feb. 2016

## Highlights & Drawbacks
The article mentioned in the experimental part that the Inception network can be upgraded to the SOTA level without the help of the Skip-connection structure, but adding Skip-connection can effectively increase the training speed.

<!-- more -->

## Motivation & Design

In 2015, ResNet became the most dazzling convolutional network structure of the year, and the structure of skip-connection became an option to avoid. The Inception series also updated its structure with reference to ResNet. At the same time, the fourth generation and the combination with ResNet were introduced: Inception-v4 and Inception-ResNet.


### Inception-v4 Architecture

[NetScope Visualization](http://ethereon.github.io/netscope/#gist/e0ac64013b167844053184d97b380978) and source code: [awesome_cnn](https://github.com/ddlee96/awesome_cnn)

![Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://i.imgur.com/VHrRqOj.jpg)

### Inception-ResNet(v2) Architecture

[NetScope Visualization](http://ethereon.github.io/netscope/#gist/aadd97383baccabb8b827ba507c24162) and source code: [awesome_cnn](https://github.com/ddlee96/awesome_cnn)

![Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://i.imgur.com/zHbsvqy.jpg)
