---
title: "CVPR 2020: Image Synthesis"
tag:
- Image Synthesis
- GAN
---

## Semantically Multi-modal Image Synthesis
- Author: Zeping Zhu, Zhi-liang Xu, Ansheng You, Xiang Bai
- Arxiv: [2003.12697](https://arxiv.org/abs/2003.12697.pdf)
- [GitHub](https://github.com/Seanseattle/SMIS)

### **Problem**
Semantically multi-modal image synthesis (SMIS): generating multi-modal images at the semantic level.


### **Assumption in prior work**
Previous work seeks to use multiple class-specific generators, constraining its usage in datasets with a small number of classes.

Gu et al.(CVPR 2019) focused on portrait editing. However, this type of methods soon face degradation in performance, a linear increase of training time and computational resource consumption under a growing number of classes. (really weak)

### **Insight**
the key is to divide the latent code into a series of class-specific latent codes each of which controls only a specific semantic class generation.


### **Technical overview**
![](https://i.imgur.com/dUXFXwj.jpg)


### **Proof**
- Datasets: DeepFashion, cityscapes, ADE20k
- Metrics: new mCSD and mOCD(based on LPIPS), FID, pixel accuracy, mIOU
- Baselines: SPADE, BicycleGAN, DSCGAN, pix2pixHD


### **Impact**
Applications: Semantically multi-modal image synthesis, Appearance mixture, Semantic manipulation, Style morphing

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



## Semantic Pyramid for Image Generation

- Author: Assaf Shocher, Yossi Gandelsman, Inbar Mosseri, Michal Yarom, Michal Irani, William T. Freeman, Tali Dekel
- Arxiv: [2003.06221](https://arxiv.org/abs/2003.06221.pdf)
- [Project site](https://semantic-pyramid.github.io)

![](https://i.imgur.com/oFgAYt7.jpg)

### **Problem**: 
Controllable Image Synthesis

### **Assumption in prior work**
The process of working in feature space typically involves the following stages: an image is fed to a pre-trained classification network; its feature responses from different layers are extracted, and optionally manipulated according to the application at hand. The manipulated features are then inverted back to an image by solving a reconstruction optimization problem. However, the problem of inverting deep features into a realistic image is challenging – there is no one-to-one mapping between the deep features and an image, especially when the features are taken from deep layers. This has been addressed so far mostly by imposing regularization priors on the reconstructed image, which often leads to blurry unrealistic reconstructions and limits the type of features that can be used.


### **Insight**
A hierarchical framework which leverages the continuum of semantic information encapsulated in such deep features; this ranges from low level information contained in fine features to high level, semantic information contained in deeper features.
By doing so, we bridge the gap between optimization based methods for feature inversion and generative adversarial learning.

### **Technical overview**
![](https://i.imgur.com/KaegNrt.jpg)

Semantic pyramid image pipeline. (a) The generator works in full mirror-like conjunction with a pre-trained classification model. Each stage of the classification model has a corresponding block in the generator. (b) Specification of a single generator block. the feature map is first multiplied by its input mask. The masked feature map then undergoes a convolution layer and the result is summed with the result of the corresponding generator block.


### **Proof**
- Datasets: Places365, Web images
- Baselines: None(Why no baseline methods?)
- Metrics: FID, Paired & Unpaired AMT test


### **Impact**
generating images with a controllable extent of semantic similarity to a reference image, and different manipulation tasks such as semantically-controlled inpainting and compositing


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



## BachGAN: High-Resolution Image Synthesis from Salient Object Layout

- Author: Yandong Li, Yu Cheng, Zhe Gan, Licheng Yu, Liqiang Wang, Jing-jing Liu
- Arxiv: [2003.11690](https://arxiv.org/abs/2003.11690.pdf)


### **Problem**
High-quality image synthesis from salient object layout.

I don't think this is a new task.

![](https://i.imgur.com/PCSOF1X.jpg)
Top row: images synthesized from semantic segmentation maps. Bottom row: high-resolution images synthesized from salient object layouts, which allows users to create an image by drawing only a few bounding boxes.

### **Assumption in prior work**
Scene graph (Johnson et al. CVPR 2018), with rich structural representation, can potentially reveal more visual relations of objects in an image. However, pairwise object relation labels are difficult to obtain in real-life applications. The lack of object size, location and background information also limits the quality of synthesized images.

Layout2im (CVPR 2019) proposed the task of image synthesis from object layout; however, both foreground and background object layouts are required, and only low-resolution images are generated.


### **Insight**
High-resolution(?) synthesis and background inference from foreground layout.


### **Technical overview**
BachGAN generates an image via two steps: (i) a background retrieval module selects from a large candidate pool a set of segmentation maps most relevant to the given object layout; (ii) these candidate layouts are then encoded via a background fusion module to hallucinate a best-matching background. With this retrieval-and-hallucination approach, BachGAN can dynamically provide detailed and realistic background that aligns well with any given foreground layout.

![](https://i.imgur.com/xj60WOu.jpg)

### **Proof**

- Datasets: cityscapes, ADE20k
- Baselines: SPADE, Layout2im
- Metrics: pixel accuracy, FID

### **Impact**
a little bit high-resolution version of Layout2im, paper-level progress


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



## Towards Unsupervised Learning of Generative Models for 3D Controllable Image Synthesis

- Author: Yiyi Liao, Katja Schwarz, Lars M. Mescheder, Andreas Geiger
- Arxiv: [1912.05237](https://arxiv.org/abs/1912.05237.pdf)
- [GitHub](https://github.com/autonomousvision/controllable image synthesis)

### **Problem**: 
![](https://i.imgur.com/SYH1qhS.jpg)

3D Controllable Image Synthesis: We define this task as an unsupervised learning problem, where a 3D controllable generative image synthesis model that allows for manipulating 3D scene properties is learned without 3D supervision.

### **Assumption in prior work**
Current image synthesis models operate in the 2D domain where disentangling 3D properties such as camera viewpoint or object pose is challenging. Furthermore, they lack an interpretable and controllable representation.


### **Insight**
Our key idea is to learn the image generation process jointly in 3D and 2D space by combining a 3D generator with a differentiable renderer and a 2D image synthesis model. This allows our model to learn abstract 3D representations which conform to the physical image formation process, thereby retaining interpretability and controllability.


### **Technical overview**
![](https://i.imgur.com/k3UsMzL.jpg)



### **Proof**
- Datasets: ShapeNet, Structured3D
- Baselines: Vanilla GAN, Layout2Im
- Metrics: FID
![](https://i.imgur.com/9JQooys.jpg)

### **Impact**
a 3D image generation baseline


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



## Image2StyleGAN++: How to Edit the Embedded Images?
- Author: Rameen Abdal, Yipeng Qin, Peter Wonka
- Arxiv: [1911.11544](https://arxiv.org/abs/1911.11544.pdf)


### **Problem**: 
Latent space editting for image synthesis


### **Technical overview**
First, we introduce noise optimization as a complement to the W+  latent space embedding.
Second, we extend the global W + latent space embedding to enable local embeddings. 
Third, we combine embedding with activation tensor manipulation to perform high quality local edits along with global semantic edits on images.


### **Impact**
some fancy face editing results
![](https://i.imgur.com/FN177Vn.jpg)


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

## Related

- [ICCV 2019: Image Synthesis(Part One)](https://arxivnote.ddlee.cn/2019/10/30/Image-Synthesis-Generation-ICCV-2019.html)
- [ICCV 2019: Image Synthesis(Part Two)](https://arxivnote.ddlee.cn/2019/10/30/Image-Synthesis-Generation-ICCV-2019-2.html)
- [ICCV 2019: Image-to-Image Translation](https://arxivnote.ddlee.cn/2019/10/24/Image-to-Image-Translation-ICCV-2019.html)
- [ICCV 2019: Face Editing and Manipulation](https://arxivnote.ddlee.cn/2019/10/29/Face-Editing-Manipulation-ICCV-2019.html)
- [GANs for Image Generation: ProGAN, SAGAN, BigGAN, StyleGAN](https://cvnote.ddlee.cn/2019/09/15/ProGAN-SAGAN-BigGAN-StyleGAN.html)
- [Deep Generative Models(Part 3): GANs(from GAN to BigGAN)](https://arxivnote.ddlee.cn/2019/08/20/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/2019/08/19/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/2019/08/18/Deep-Generative-Models-Taxonomy-VAE.html)