---
title: "U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation - Junho Kim - 2019"
tag:
- Image-to-Image Translation
- GAN
redirect_from: /U-GAT-IT-Unsupervised-Generative-Attentional-Networks-with-Adaptive-Layer-Instance-Normalization-for-Image-to-Image-Translation-Junho-Kim-2019.html
---



## Info
- Title: **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**
- Task: **Image-to-Image Translation**
- Author: Junho Kim, Minjae Kim, Hyeonwoo Kang, Kwanghee Le
- Date:  July. 2019
- Arxiv: [1907.10830](https://arxiv.org/abs/1907.10830)

## Highlights & Drawbacks
Where earlier image-to-image translation networks work best with particular image styles, U-GAT-IT adds layers that make it useful across a variety of styles.
Such networks typically represent shapes and textures in hidden feature maps. U-GAT-IT adds a layer that weights the importance of each feature map based on each image’s style. 
The researchers also introduce a layer that learns which normalization method works best.


## Abstract 
 We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus on more important regions distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. Experimental results show the superiority of the proposed method compared to the existing state-of-the-art models with a fixed network architecture and hyper-parameters.

## Motivation & Design

Image-to-image translation, in which stylistic features from one image are imposed on the content of another to create a new picture, traditionally has been limited to translating either shapes or textures. A new network translates both, allowing more flexible image combinations and creating more visually satisfying output.

![U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation - Junho Kim - 2019](https://i.imgur.com/KOGaUjw.jpg)

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

U-GAT-IT uses a typical GAN architecture: A discriminator classifies images as either real or generated and a generator tries to fool the discriminator. It accepts two image inputs.
The generator takes the images and uses a CNN to extract feature maps that encode shapes and textures. 

In earlier models, feature maps are passed directly to an attention layer that models the correspondence between pixels in each image. In U-GAT-IT, an intermediate weighting layer learns the importance of each feature map. The weights allow the system to distinguish the importance of different textures and shapes in each style.

The weighted feature maps are passed to the attention layer to assess pixel correspondences, and the generator produces an image from there.
The discriminator takes the first image as a real-world style example and the second as a candidate in the same style that’s either real or generated.
Like the generator, it encodes both images to feature maps via a CNN and uses a weighting layer to guide an attention layer. 
The discriminator classifies the candidate image based on the attention layer's output.


## Performance & Ablation Study
 Test subjects chose their favorite images from a selection of translations by U-GAT-IT and four earlier methods. The subjects preferred U-GAT-IT’s output by up to 73% in four out of five data sets.

 ![U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation - Junho Kim - 2019](https://i.imgur.com/I1cs3Ji.jpg)


 Visual comparisons on the five datasets. From top to bottom: selfie2anime, horse2zebra, cat2dog, photo2portrait, and photo2vangogh. (a) Source images, (b) U-GAT-IT, (c) CycleGAN, (d) UNIT, (e) MUNIT, (f) DRIT.

 ![U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation - Junho Kim - 2019](https://i.imgur.com/NzTS1My.jpg)

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
- [TensorFlow](https://github.com/taki0112/UGATIT)
- [PyTorch](https://github.com/znxlwm/UGATIT-pytorch)



## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
