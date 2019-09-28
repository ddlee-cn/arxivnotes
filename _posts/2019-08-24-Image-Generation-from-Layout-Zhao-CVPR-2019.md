---
title: Image Generation from Layout - Zhao - CVPR 2019
tag:
- Image Generation
- GAN
redirect_from: /Image-Generation-from-Layout-Zhao-CVPR-2019.html
---





## Info
- Title: **Image Generation from Layout**
- Task: **Image Generation**
- Author: Bo Zhao Lili Meng Weidong Yin Leonid Sigal
- Date:  Nov. 2018
- Arxiv: [1811.11389](http://arxiv.org/abs/1811.11389)
- Published: CVPR 2019

## Abstract

Despite significant recent progress on generative models, controlled generation of images depicting multiple and complex object layouts is still a difficult problem. Among the core challenges are the diversity of appearance a given object may possess and, as a result, exponential set of images consistent with a specified layout. To address these challenges, we propose a novel approach for layout-based image generation; we call it Layout2Im. Given the coarse spatial layout (bounding boxes + object categories), our model can generate a set of realistic images which have the correct objects in the desired locations. The representation of each object is disentangled into a specified/certain part (category) and an unspecified/uncertain part (appearance). The category is encoded using a word embedding and the appearance is distilled into a low-dimensional vector sampled from a normal distribution. Individual object representations are composed together using convolutional LSTM, to obtain an encoding of the complete layout, and then decoded to an image. Several loss terms are introduced to encourage accurate and diverse generation. The proposed Layout2Im model significantly outperforms the previous state of the art, boosting the best reported inception score by 24.66% and 28.57% on the very challenging COCO-Stuff and Visual Genome datasets, respectively. Extensive experiments also demonstrate our method's ability to generate complex and diverse images with multiple objects.

## Highlights

A novel approach for generating images from coarse layout (bounding boxes + object categories). This provides a flexible control mechanism for image generation.

By disentangling the representation of objects into a category and (sampled) appearance, the proposed model is capable of generating a diverse set of consistent images from the same layout.

## Motivation & Design

![Image Generation from Layout - Zhao - CVPR 2019](https://i.imgur.com/vc9XgJY.png)

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

Image generation from layout. Given the coarse layout (bounding boxes + object categories), the proposed Layout2Im model samples the appearance of each object from a normal distribution, and transforms these inputs into a real image by a serial of components. 

![Image Generation from Layout - Zhao - CVPR 2019](https://i.imgur.com/RZHLIgT.png)

Overview of Layout2Im network for generating images from layout during training. 

The inputs to the model are the ground truth image with its layout. The objects are first cropped from the input image according to their bounding boxes, and then processed with the object estimator to predict a latent code for each object. After that, multiple object feature maps are prepared by the object composer based on the latent codes and layout, and processed with the object encoder, objects fuser and image decoder to reconstruct the input image. Additional set of latent codes are also sampled from a normal distribution to generate a new image. Finally, objects in generated images are used to regress the sampled latent codes. The model is trained adversarially against a pair of discriminators and a number of objectives.
![Image Generation from Layout - Zhao - CVPR 2019](https://i.imgur.com/Hz2E9JR.png)

Object latent code estimation. Given the input image and its layout, the objects are first cropped and resized from the input image. Then the object estimator predicts a distribution for each object from the object crops, and multiple latent codes are sampled from the estimated distribution.
![Image Generation from Layout - Zhao - CVPR 2019](https://i.imgur.com/BQ4VKHh.png)

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

Object feature map composition

The object category is fiirs encoded by a word embedding Then the objec feature map is simply composed by filling the regionwithin the object bounding box with the concatenation of category embedding and latent code. The rest of the feature map are alll zeors. 

Loss functions
![Image Generation from Layout - Zhao - CVPR 2019](https://i.imgur.com/zPqg0Ka.png)

![Image Generation from Layout - Zhao - CVPR 2019](https://i.imgur.com/V1J1wIb.png)

## Performance & Ablation Study

Qualitative Results
![](https://i.imgur.com/o52IKYh.png)

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

Quantitative Results
![Image Generation from Layout - Zhao - CVPR 2019](https://i.imgur.com/kYMS1wg.png)

![Image Generation from Layout - Zhao - CVPR 2019](https://i.imgur.com/J5nfTDQ.png)

## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
