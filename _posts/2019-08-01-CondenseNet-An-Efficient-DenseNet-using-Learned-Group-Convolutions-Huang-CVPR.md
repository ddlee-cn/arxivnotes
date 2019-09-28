---
title: "CondenseNet: An Efficient DenseNet using Learned Group Convolutions - Huang - CVPR 2018"
tag:
- Image Classification
- CNN
redirect_from: /CondenseNet-An-Efficient-DenseNet-using-Learned-Group-Convolutions-Huang-CVPR.html
---



## Info
- Title: **CondenseNet: An Efficient DenseNet using Learned Group Convolutions**
- Task: **Image Classification**
- Author: Gao Huang, Shichen Liu, Laurens van der Maaten, Kilian Q. Weinberger
- Date: Nov. 2017
- Arxiv: [1711.09224](https://arxiv.org/abs/1711.09224)
- Published: CVPR 2018

## Highlights & Drawbacks
- Learned manner for group hyper-params
- Implementation with standard grouped convolutions


## Motivation & Design
**Group convolution**
![CleanShot 2019-08-18 at 11.25.54@2x](media/15598791313598/CleanShot%202019-08-18%20at%2011.25.54@2x.jpg)
Standard convolution (left) and group convolution (right). The latter enforces a sparsity pattern by partitioning the inputs (and outputs) into disjoint groups

![CleanShot 2019-08-18 at 11.27.49@2x](media/15598791313598/CleanShot%202019-08-18%20at%2011.27.49@2x.jpg)
Illustration of learned group convolutions with G = 3 groups and a condensation factor of C = 3. During training a fraction of (C − 1)/C connections are removed after each of the C − 1 condensing stages. Filters from the same group use the same set of features, and during test-time the index layer rearranges the features to allow the resulting model to be implemented as standard group convolutions.

## Performance & Ablation Study
Ablation study on CIFAR-10 to investigate the efficiency gains obtained by the various components of CondenseNet.
![CleanShot 2019-08-18 at 11.28.28@2x](media/15598791313598/CleanShot%202019-08-18%20at%2011.28.28@2x.jpg)

![CleanShot 2019-08-18 at 11.29.01@2x](media/15598791313598/CleanShot%202019-08-18%20at%2011.29.01@2x.jpg)
Actual inference time of different models on an ARM processor. All models are trained on ImageNet, and accept input with resolution 224×224.


## Code
[PyTorch](https://github.com/ShichenLiu/CondenseNet)