---
title: Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017
tag:
- GAN
- Image Inpainting
redirect_from: Globally-and-locally-consistent-image-completion-SIGGRAPH.html
---



## Info
- Title: **Globally and locally consistent image completion**
- Task: **Image Inpainting**
- Author: S. Iizuka, E. Simo-Serra, and H. Ishikawa
- Date:  Jul. 2017
- Published: SIGGRAPH 2017

## Highlights & Drawbacks
- A high performance network model that can complete arbitrary
missing regions
- A globally and locally consistent adversarial training approach
for image completion

## Motivation & Design

![Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://i.imgur.com/8vc75lO.jpg)

**Overview of network architecture**
It consists of a completion network and two auxiliary context discriminator networks that are used only for training the completion network and are not used during the testing. The global discriminator network takes the entire image as input, while the local discriminator network takes only a small region around the completed area as input. Both discriminator networks are trained to determine if an image is real or completed by the completion network, while the completion network is trained to fool both discriminator networks.

![Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://i.imgur.com/RbL82jP.jpg)

**Loss Design**

Weighted MSE loss is defined by,
$$
L\left(x, M_{c}\right)=\left\|M_{c} \odot\left(C\left(x, M_{c}\right)-x\right)\right\|^{2}
$$
where $⊙$ is the pixel-wise multiplication and $∥ · ∥$ is the Euclidean norm.

The global and local discriminator network also work as GAN Loss:
$$
\min _{C} \max _{D} \mathbb{E}\left[\log D\left(x, M_{d}\right)+\log \left(1-D\left(C\left(x, M_{c}\right), M_{c}\right)\right]\right.
$$
where $M_d$ is a random mask, $M_c$ is the input mask, and the expectation value is just the average over the training images $x$.

Combine above, the optimization becomes:

$$
\begin{array}{rl}{\min _{C} \max _{D}} & {\mathbb{E}\left[L\left(x, M_{c}\right)+\alpha \log D\left(x, M_{d}\right)\right.} \\ {} & {\left.+\alpha \log \left(1-D\left(C\left(x, M_{c}\right), M_{c}\right)\right)\right]}\end{array}
$$


## Performance & Ablation Study
![Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://i.imgur.com/4AwGRTj.jpg)

We compare with Photoshop Content Aware Fill(PatchMatch), ImageMelding,[Huangetal.2014], and Context Encoder[Pathaketal. 2016] using random masks. For the comparison, we have retrained the model of [Pathak et al. 2016] on the Places2 dataset for arbitrary region completion. Furthermore, we use the same post-processing as used for our approach. We can see that, while PatchMatch and Image Melding generate locally consistent patches extracted from other parts of the image, they are not globally consistent with the other parts of the scene. The approach of [Pathak et al. 2016] can inpaint novel regions, but the inpainted region tends to be easy to identify, even with our post-processing. Our approach, designed to be both locally and globally consistent, results in much more natural scenes. 

![Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://i.imgur.com/XW1Y6x2.jpg)

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

## Code
- [Torch(Official)](https://github.com/satoshiiizuka/siggraph2017_inpainting)
- [PyTorch](https://github.com/otenim/GLCIC-PyTorch)

## Related
- [Image Inpainting: From PatchMatch to Pluralistic](https://arxivnote.ddlee.cn/Imbalance-Problems-in-Object-Detection-A-Review-Oksuz-2019.html)
- [Generative Image Inpainting with Contextual Attention - Yu - CVPR 2018 - TensorFlow](https://arxivnote.ddlee.cn/Generative-Image-Inpainting-with-Contextual-Attention-Yu-CVPR-TensorFlow.html)
- [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning - Nazeri - 2019 - PyTorch](https://arxivnote.ddlee.cn/EdgeConnect-Generative-Image-Inpainting-with-Adversarial-Edge-Learning-Nazeri.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)