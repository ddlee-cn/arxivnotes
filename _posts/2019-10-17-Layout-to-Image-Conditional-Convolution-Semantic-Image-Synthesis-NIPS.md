---
title: Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis
tag:
- Image Generation
- Conditioned Image Synthesis
---

## Info

- Title: Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis
- Task: Image Generation
- Date: Oct. 2019
- Arxiv: [1910.06809](https://arxiv.org/abs/1910.06809)
- Published: NIPS 2019


## Abstract

Semantic image synthesis aims at generating photorealistic images from semantic layouts. Previous approaches with conditional generative adversarial networks (GAN) show state-of-the-art performance on this task, which either feed the semantic label maps as inputs to the generator, or use them to modulate the activations in normalization layers via affine transformations. We argue that convolutional kernels in the generator should be aware of the distinct semantic labels at different locations when generating images. In order to better exploit the semantic layout for the image generator, we propose to predict convolutional kernels conditioned on the semantic label map to generate the intermediate feature maps from the noise maps and eventually generate the images. Moreover, we propose a feature pyramid semantics-embedding discriminator, which is more effective in enhancing fine details and semantic alignments between the generated images and the input semantic layouts than previous multi-scale discriminators. We achieve state-of-the-art results on both quantitative metrics and subjective evaluation on various semantic segmentation datasets, demonstrating the effectiveness of our approach.


## Motivation & Design


![Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis](https://i.imgur.com/UB1PZkY.png)

(Left) The structure of a Conditional Convolution Block (CC Block). (Right) The overall
framework of our proposed CC-FPSE. The weight prediction network predicts weights for CC Blocks in the generator. The conditional convolution generator is built up of Conditional Convolution (CC) Blocks shown on the left. The feature pyramid semantics-embedding (FPSE) discriminator predicts real/fake scores as well as semantic alignment scores. L-ReLU in the CC Block denotes Leaky ReLU.

### Conditional Convolution

To mitigate the problem, SPADE uses the label maps to predict spatially-adaptive affine transformations for modulating the activations in normalization layers. However, such feature modulation by simple affine transformations is limited in representational power and flexibility.


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




On the other hand, we rethink the functionality of convolutional layers for image synthesis. In a generation network, each convolutional layer learns “how to draw” by generating fine features at each location based on a local neighborhood of input features. The same translation-invariant convolutional kernels are applied to all samples and at all spatial locations, irrespective of different semantic labels at different locations, as well as the unique semantic layout of each sample. Our argument is that different convolutional kernels should be used for generating different objects or stuff.

$$
\mathbf{Y}_{c, i, j}=\sum_{m=0}^{k-1} \sum_{n=0}^{k-1} \mathbf{X}_{c, i+m, j+n} \mathbf{V}_{c, m, n, i, j}
$$

where i, j denotes the spatial coordinates of the feature maps, k denotes the convolution kernel size, and c denotes the channel index. The C × H × W convolutional kernels in V with kernel size k × k operates at each channel and each spatial location of X independently to generate output feature maps.


### Feature Pyramid Semantics-Embedding Discriminator

The authors believe that a robust discriminator should focus on two indispensable and complementary aspects of the images: high-fidelity details, and semantic alignment with the input layout map. Motivated by the two principles, we propose to utilize multi-scale feature pyramids for promoting high-fidelity details such as texture and edges, and exploit patch-based semantic-embeddings to enhance the spatial semantic alignment between the generated images and the input semantic layout.

The proposed  feature pyramid discriminator takes the input image at a single scale. The bottom-up pathway produces a feature hierarchy consisting of multi-scale feature maps and the top-down pathway gradually upsamples the spatially coarse but semantically rich feature maps. The lateral combines the high-level semantic feature maps from the top-down pathway with the low-level feature maps from the bottom-up pathway.


## Experiments & Ablation Study

![Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis](https://i.imgur.com/QycdaZK.png)


## Code

[PyTorch](https://github.com/xh-liu/CC-FPSE)

## Related

- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/2019/08/21/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE and INIT](https://arxivnote.ddlee.cn/2019/08/22/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [(DMIT)Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://arxivnote.ddlee.cn/2019/10/08/Multi-mappitng-Image-to-Image-Translation-Disentanglement.html)
- [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation - Junho Kim - 2019](https://arxivnote.ddlee.cn/2019/09/18/U-GAT-IT-Unsupervised-Generative-Attentional-Networks-with-Adaptive-Layer-Instance-Normalization-for-Image-to-Image-Translation-Junho-Kim-2019.html)
- [U-GAT-IT: TensorFlow Implementation](https://cvnote.ddlee.cn/2019/09/10/U-GAT-IT-TensorFlow.html)
- [U-GAT-IT: PyTorch Implementation](https://cvnote.ddlee.cn/2019/09/10/U-GAT-IT-PyTorch.html)
- [Towards Instance-level Image-to-Image Translation - Shen - CVPR 2019](https://arxivnote.ddlee.cn/2019/07/18/Towards-Instance-level-Image-to-Image-Translation-Shen-CVPR-2019.html)

- [PyTorch Code for SPADE](https://cvnote.ddlee.cn/2019/09/14/SPADE-PyTorch-GitHub.html)

