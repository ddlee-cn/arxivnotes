---
title: Few-shot Video-to-Video Synthesis
tag:
- Video Translation
- Few-Shot Learning
---

## Info

- Title: Few-shot Video-to-Video Synthesis
- Task: Video-to-Video Translation
- Author: Ting-Chun Wang, Ming-Yu Liu, Andrew Tao, Guilin Liu, Jan Kautz, Bryan Catanzaro
- Date: Oct. 2019
- Published: NIPS 2019


## Abstract

Video-to-video synthesis (vid2vid) aims at converting an input semantic video, such as videos of human poses or segmentation masks, to an output photorealistic video. While the state-of-the-art of vid2vid has advanced signiﬁcantly, existing approaches share two major limitations. First, they are data-hungry. Numerous images of a target human subject or a scene are required for training. Second, a learned model has limited generalization capability. A pose-to-human vid2vid model can only synthesize poses of the single person in the training set. It does not generalize to other humans that are not in the training set. To address the limitations, we propose a few-shot vid2vid framework, which learns to synthesize videos of previously unseen subjects or scenes by leveraging few example images of the target at test time. Our model achieves this few-shot generalization capability via a novel network weight generation module utilizing an attention mechanism. We conduct extensive experimental validations with comparisons to strong baselines using several large-scale video datasets including human-dancing videos, talking-head videos, and street-scene videos. The experimental results verify the effectiveness of the proposed framework in addressing the two limitations of existing vid2vid approaches.



## Motivation & Design

![](https://i.imgur.com/ZXnlo03.png)

Comparison between the vid2vid (left) and the proposed few-shot vid2vid (right).

Existing vid2vid methods [7, 12, 57] do not consider generalization to unseen domains. A trained model can only be used to synthesize videos similar to those in the training set. For example, a vid2vid model can only be used to generate videos of the person in the training set. To synthesize a new person, one needs to collect a dataset of the new person and uses it to train a new vid2vid model. On the other hand, our few-shot vid2vid model does not have the limitations. Our model can synthesize videos of new persons by leveraging few example images provided at the test time.


### Formulation

F take two more input arguments: one is a set of K example images {e1 , e2, ..., eK } of the target domain, and the other is the set of their corresponding semantic images {se1 , se2 , ..., seK }. That is

$$
\tilde{\mathbf{x}}_{t}=F\left(\tilde{\mathbf{x}}_{t-\tau}^{t-1}, \mathbf{s}_{t-\tau}^{t},\left\{\mathbf{e}_{1}, \mathbf{e}_{2}, \ldots, \mathbf{e}_{K}\right\},\left\{\mathbf{s}_{\mathbf{e}_{1}}, \mathbf{s}_{\mathbf{e}_{2}}, \ldots, \mathbf{s}_{\mathbf{e}_{K}}\right\}\right)
$$

This modeling allows F to leverage the example images given at the test time to extract some useful patterns to synthesize videos of the unseen domain. We propose a network weight generation module E for extracting the patterns. Specifically, E is designed to extract patterns from the provided example images and use them to compute network weights θH for the intermediate image synthesis network H:

$$
\boldsymbol{\theta}_{H}=E\left(\tilde{\mathbf{x}}_{t-\tau}^{t-1}, \mathbf{s}_{t-\tau}^{t},\left\{\mathbf{e}_{1}, \mathbf{e}_{2}, \ldots, \mathbf{e}_{K}\right\},\left\{\mathbf{s}_{\mathbf{e}_{1}}, \mathbf{s}_{\mathbf{e}_{2}}, \ldots, \mathbf{s}_{\mathbf{e}_{K}}\right\}\right)
$$

### Overall Architecture



![](https://i.imgur.com/jg6BOXn.png)



(a) Architecture of the vid2vid framework. (b) Architecture of the proposed few-shot vid2vid framework. It consists of a network weight generation module $E$ that maps example images to part of the network weights for video synthesis. The module $E$ consists of three sub-networks: $E_F$ ,$E_P$ , and $E_A$ (used when K > 1). The sub-network $E_F$ extracts features q from the example images. When there are multiple example images (K > 1), $E_A$ combines the extracted features by estimating soft attention maps α and weighted averaging different extracted features. The final representation is then fed into the network $E_P$ to generate the weights $\theta_H$ for the image synthesis network $H$.



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


### Network Weight Generation

We decompose E into two sub-networks: an example feature extractor $E_F$ , and a multi-layer
perceptron $E_P$ . The network $E_F$ consists of several convolutional layers and is applied on the
example image e1 to extract an appearance representation q. The representation q is then fed into
$E_P$ to generate the weights $θ_H$ in the intermediate image synthesis network H.





### Attention-based Aggregation

$E_A$ is applied to each of the semantic images of the example images, sek . This results in
a key vector $a_k ∈ R_{C×N}$ , where C is the number of channels and N = H × W is the spatial
dimension of the feature map. We also apply $E_A$ to the current input semantic image st to extract its
key vector $a_t ∈ R_{C×N}$ . We then compute the attention weight $α_k ∈ R{N ×N}$ by taking the matrix
product $α_k = (a_k )^T ⊗ a_t$ . The attention weights are then used to compute a weighted average of the
appearance representation $q = \Sigma^{K}_{k=1} q_k ⊗ α_k$ , which is then fed into the multi-layer perceptron EP
to generate the network weights.






## Experiments & Ablation Study



The model results are available at the [Project Site](https://nvlabs.github.io/few-shot-vid2vid/).




## Code

[Project Site](https://nvlabs.github.io/few-shot-vid2vid/)

[PyTorch](https://github.com/NVlabs/few-shot-vid2vid)



## Related

- [Few-Shot Unsupervised Image-to-Image Translation](https://arxivnote.ddlee.cn/2019/10/27/Few-Shot-Unsupervised-Image-to-Image-Translation-ICCV.html)

- [Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen - ACM MM 2019](https://arxivnote.ddlee.cn/2019/09/20/Mocycle-GAN-Unpaired-Video-to-Video-Translation-Yang-Chen-ACMMM-2019.html)
- [Recycle-GAN: Unsupervised Video Retargeting - Bansal - ECCV 2018](https://arxivnote.ddlee.cn/2019/09/17/Recycle-GAN-Unsupervised-Video-Retargeting-Bansal-ECCV-2018.html)
- [mage to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/2019/08/21/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE and INIT](https://arxivnote.ddlee.cn/2019/08/22/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)

- [PyTorch Code for vid2vid](https://cvnote.ddlee.cn/2019/09/06/vid2vid-PyTorch-GitHub.html)

- [PyTorch Code for SPADE](https://cvnote.ddlee.cn/2019/09/14/SPADE-PyTorch-GitHub.html)

