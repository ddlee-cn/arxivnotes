---
title: "TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation - Wayne Wu - CVPR 2019"
tag:
- Image-to-Image Translation
- GAN
redirect_from: /TransGaGa-Geometry-Aware-Unsupervised-Image-to-Image-Translation-Wayne-Wu-CVPR-2019.html
---



## Info

- Title: TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation
- Task: Image-to-Image Translation
- Author: Wayne Wu, Kaidi Cao, Cheng Li, Chen Qian, Chen Change Loy
- Arxiv: [1904.09571](https://arxiv.org/abs/1904.09571)
- Published: CVPR 2019


## Abstract

Unsupervised image-to-image translation aims at learning a mapping between two visual domains. However, learning a translation across large geometry variations always ends up with failure. In this work, we present a novel disentangle-and-translate framework to tackle the complex objects image-to-image translation task. Instead of learning the mapping on the image space directly, we disentangle image space into a Cartesian product of the appearance and the geometry latent spaces. Specifically, we first introduce a geometry prior loss and a conditional VAE loss to encourage the network to learn independent but complementary representations. The translation is then built on appearance and geometry space separately. Extensive experiments demonstrate the superior performance of our method to other state-of-the-art approaches, especially in the challenging near-rigid and non-rigid objects translation tasks. In addition, by taking different exemplars as the appearance references, our method also supports multimodal translation. Project page: [https://wywu.github.io/projects/TGaGa/TGaGa.html](https://wywu.github.io/projects/TGaGa/TGaGa.html)

## Motivation & Design
![TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation - Wayne Wu - CVPR 2019](https://i.imgur.com/XWB3EtS.png)

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

The proposed framework consists of four main components: two auto-encoders (X/Y domain) and two transformers (geometry/appearance). 

Auto-Encoder: Taking X domain for example. For the input x, we use an encoder Ex to obtain the
geometry representation gx , which is a 30-channel point-heatmap with the same resolution as x. We project all channels of gx together for visualisation. Then, gx is embedded again to get the geometry code cx . At the same time, x is also embedded by appearance encoder Ex to get the appearance code ax . Finally, ax  and cx are concatenated together to generate x̂ with Dx. 

Transformer: For cross-domain translation, geometry (gx ↔ gy ) and appearance (ax ↔ ay ) transformation are performed separately.

The appearance transformer:

![TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation - Wayne Wu - CVPR 2019](https://i.imgur.com/05CWyY3.png)


The Geometry Transformer:

![TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation - Wayne Wu - CVPR 2019](https://i.imgur.com/GXwB42B.png)

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

To incorporate the PCA landmark representation with GAN, we replace all Conv-ReLU blocks with FC-ReLU blocks in both generators and discriminators. Though we incorporate a similar transformer structure as in CariGANs [4], our work differs in that unlike CariGANs that uses landmarks’ PCA embeddings directly as the source and target domain defined in CycleGAN, we train the corresponding cycle on image pixel level which is more direct and powerful for pose-preserving generation task.


## Experiments & Ablation Study

![TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation - Wayne Wu - CVPR 2019](https://i.imgur.com/3svPF15.png)


![TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation - Wayne Wu - CVPR 2019](https://i.imgur.com/2zdJzAm.png)

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

![TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation - Wayne Wu - CVPR 2019](https://i.imgur.com/EuZDQvC.png)

## Code

[Project Site]([https://wywu.github.io/projects/TGaGa/TGaGa.html](https://wywu.github.io/projects/TGaGa/TGaGa.html))

## Related

- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
