---
title: Bag of Tricks for Image Classification with Convolutional Neural Networks - He - 2018
tag:
- Image Classification
- Engineering
redirect_from: /Bag-of-Tricks-for-Image-Classification-with-Convolutional-Neural-Networks-He-2018.html
---



## Info

- Title: **Bag of Tricks for Image Classification with Convolutional Neural Networks**
- Task: Image Classification
- Author: Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
- Arxiv: [1812.01187](https://arxiv.org/abs/1812.01187)

## Abstract

Much of the recent progress made in image classification research can be credited to training procedure refinements, such as changes in data augmentations and optimization methods. In the literature, however, most refinements are either briefly mentioned as implementation details or only visible in source code. In this paper, we will examine a collection of such refinements and empirically evaluate their impact on the final model accuracy through ablation study. We will show that, by combining these refinements together, we are able to improve various CNN models significantly. For example, we raise ResNet-50's top-1 validation accuracy from 75.3% to 79.29% on ImageNet. We will also demonstrate that improvement on image classification accuracy leads to better transfer learning performance in other application domains such as object detection and semantic segmentation.

![](https://i.imgur.com/ki1FXpz.png)

## Linear scaling learning rate

In mini-batch SGD, gradient descending is a random process because the examples are randomly selected in each batch. Increasing the batch size does not change the expectation of the stochastic gradient but reduces its variance. In other words, a large batch size reduces the noise in the gradient, so we may increase the learning rate to make a larger progress along the opposite of the gradient direction. Goyal et al. reports that linearly increasing the learning rate with the batch size works empirically for ResNet-50 training. In particular, if we follow He et al. to choose 0.1 as the initial learning rate for batch size 256, then when changing to a larger batch size b, we will increase the initial learning rate to 0.1 × b/256.

## Learning rate warmup

At the beginning of the training, all parameters are typically random values and therefore far away from the final solution. Using a too large learning rate may result in numerical instability. In the warmup heuristic, we use a small learning rate at the beginning and then switch back to the initial learning rate when the training process is stable Goyal et al. proposes a gradual warmup strategy that increases the learning rate from 0 to the initial learning rate linearly. In other words, assume we will use the first m batches (e.g. 5 data epochs) to warm up, and the initial learning rate is η, then at batch i, 1 ≤ i ≤ m, we will set the learning rate to be iη/m.

## No bias decay

The weight decay is often applied to all learnable parameters including both weights and bias. It’s
equivalent to applying an L2 regularization to all parameters to drive their values towards 0. As pointed out by Jia et al. [14], however, it’s recommended to only apply the regularization to weights to avoid overfitting. The no bias decay heuristic follows this recommendation, it only applies the weight decay to the weights in convolution and fully-connected layers. Other parameters, including the biases and γ and β in BN layers, are left unregularized.
![](https://i.imgur.com/KdwxV0H.png)

## Label Smoothing
![](https://i.imgur.com/CDUtAgF.png)

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
## Object Detection Results
![](https://i.imgur.com/wDGtvF6.png)


## Semantic Segmentation Results
![](https://i.imgur.com/zNhVmul.png)
