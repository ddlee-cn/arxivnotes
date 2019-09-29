---
title: "Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE and INIT"
tag:
- Image-to-Image Translation
- GAN
- Review
redirect_from: /Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html
---







## pix2pixHD - Ting-Chun Wang - CVPR 2018
- Title: **High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs**
- Author: Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, A. Tao, J. Kautz, and B. Catanzaro
- Date: Nov. 2017.
- Arxiv: [1711.11585](https://arxiv.org/abs/1711.11585)
- Published: CVPR 2018

### Coarse-to-fine generator
![pix2pixHD - Ting-Chun Wang - CVPR 2018](https://i.imgur.com/Zzd82Ui.jpg)

We first train a residual network G1 on lower resolution images. Then, an- other residual network G2 is appended to G1 and the two networks are trained jointly on high resolution images. Specifically, the input to the residual blocks in G2 is the element-wise sum of the feature map from G2 and the last feature map from G1.

### Multi-scale Discriminator
![pix2pixHD - Ting-Chun Wang - CVPR 2018](https://i.imgur.com/L3ambvG.jpg)

### Feature-Matching Loss
This loss stabilizes the training as the generator has to produce natural statistics at multiple scales. Specifically, we extract features from multiple layers of the
discriminator and learn to match these intermediate representations from the real and the synthesized image. For ease of presentation, we denote the ith-layer feature extractor of discriminator $D_k$ as $D_(i)$ (from input to the ith layer of $D_k$). The feature matching loss calculated as:
$$
\mathcal{L}_{\mathrm{FM}}\left(G, D_{k}\right)=\mathbb{E}_{(\mathbf{s}, \mathbf{x})} \sum_{i=1}^{T} \frac{1}{N_{i}}\left[\left\|D_{k}^{(i)}(\mathbf{s}, \mathbf{x})-D_{k}^{(i)}(\mathbf{s}, G(\mathbf{s}))\right\|_{1}\right]
$$

### Using semantic labels to produce boundary map
The authors argue that the most critical information the instance map provides, which is not available in the semantic label map, is the object boundary. 
![pix2pixHD - Ting-Chun Wang - CVPR 2018](https://i.imgur.com/vXdOyOe.jpg)

(a) a typical semantic la- bel map. Note that all connected cars have the same label, which makes it hard to tell them apart. (b) The extracted instance boundary map. With this information, separating different objects becomes much easier.

### Learning an instance-level feature emedding
![pix2pixHD - Ting-Chun Wang - CVPR 2018](https://i.imgur.com/JNV2eBE.jpg)

To generate diverse images and allow instance-level control, we propose adding additional low-dimensional feature channels as the input to the generator network. We show that, by manipulating these features, we can have flexible control over the image synthesis process. Furthermore, note that since the feature channels are continuous quantities, our model is, in principle, capable of generating infinitely many images.

To generate the low-dimensional features, we train an encoder network E to find a low-dimensional feature vector that corresponds to the ground truth target for each instance in the image. Our feature encoder architecture is a standard encoder-decoder network. To ensure the features are consistent within each instance, we add an instance-wise average pooling layer to the output of the encoder to compute the average feature for the object instance. The average feature is then broadcast to all the pixel locations of the instance.

### Code
[Project Site](https://tcwang0509.github.io/pix2pixHD/)
[PyTorch](https://github.com/NVIDIA/pix2pixHD)

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

## (MUNIT)Multimodal Unsupervised Image-to-Image Translation - Huang - ECCV 2018
- Title: **Multimodal Unsupervised Image-to-Image Translation**
- Author: Xun Huang, Ming-Yu Liu, S. Belongie, and J. Kautz
- Date: Apr. 2018
- Arxiv: [1804.04732](https://arxiv.org/abs/1804.04732)
- Published: ECCV 2018

### The model design
![(MUNIT)Multimodal Unsupervised Image-to-Image Translation - Huang - ECCV 2018](https://i.imgur.com/LelrVMe.jpg)

(a) Images in each domain $X_i$ are encoded to a shared content space $C$ and a domain-specific style space $S_i$. Each encoder has an inverse decoder omitted from this figure. 
(b) To translate an image in $X_1$ (e.g., a leopard) to $X_2$ (e.g., domestic cats), we recombine the content code of the input with a random style code in the target style space. Different style codes lead to different outputs.

![(MUNIT)Multimodal Unsupervised Image-to-Image Translation - Huang - ECCV 2018](https://i.imgur.com/HZBg9Ga.jpg)

The proposed model consists of two auto- encoders (denoted by red and blue arrows respectively), one for each domain. The latent code of each auto-encoder is composed of a content code c and a style code s. We train the model with adversarial objectives (dotted lines) that ensure the translated images to be indistinguishable from real images in the target domain, as well as bidirectional reconstruction objectives (dashed lines) that reconstruct both images and latent codes.

### The AutoEncoder Architecture
![(MUNIT)Multimodal Unsupervised Image-to-Image Translation - Huang - ECCV 2018](https://i.imgur.com/tm3GTLH.jpg)

The content encoder consists of several strided convolutional layers followed by residual blocks. The style encoder contains several strided convolutional layers followed by a global average pooling layer and a fully connected layer. The decoder uses a MLP to produce a set of AdaIN [54] parameters from the style code. The content code is then processed by residual blocks with AdaIN layers, and finally decoded to the image space by upsampling and convolutional layers.


### Loss design
Image reconstruction. Given an image sampled from the data distribution, we should be able to reconstruct it after encoding and decoding.
$$
\mathcal{L}_{\mathrm{recon}}^{x_{1}}=\mathbb{E}_{x_{1} \sim p\left(x_{1}\right)}\left[\left\|G_{1}\left(E_{1}^{c}\left(x_{1}\right), E_{1}^{s}\left(x_{1}\right)\right)-x_{1}\right\|_{1}\right]
$$

Latent reconstruction. Given a latent code (style and content) sampled from the latent distribution at translation time, we should be able to recon- struct it after decoding and encoding.
$$
\begin{aligned} \mathcal{L}_{\text {recon }}^{c_{1}} &=\mathbb{E}_{c_{1} \sim p\left(c_{1}\right), s_{2} \sim q\left(s_{2}\right)}\left[\left\|E_{2}^{c}\left(G_{2}\left(c_{1}, s_{2}\right)\right)-c_{1}\right\|_{1}\right] \\ \mathcal{L}_{\text {recon }}^{s_{2}} &=\mathbb{E}_{c_{1} \sim p\left(c_{1}\right), s_{2} \sim q\left(s_{2}\right)}\left[\left\|E_{2}^{s}\left(G_{2}\left(c_{1}, s_{2}\right)\right)-s_{2}\right\|_{1}\right] \end{aligned}
$$

Adversarial loss. We employ GANs to match the distribution of translated images to the target data distribution. In other words, images generated by our model should be indistinguishable from real images in the target domain.

$$
\mathcal{L}_{\mathrm{GAN}}^{x_{2}}=\mathbb{E}_{c_{1} \sim p\left(c_{1}\right), s_{2} \sim q\left(s_{2}\right)}\left[\log \left(1-D_{2}\left(G_{2}\left(c_{1}, s_{2}\right)\right)\right)\right]+\mathbb{E}_{x_{2} \sim p\left(x_{2}\right)}\left[\log D_{2}\left(x_{2}\right)\right]
$$

where $D_2$ is a discriminator that tries to distinguish between translated images and real images in $X_2$. The discriminator $D_1$ and loss $L^{x1}_{GAN}$ are defined similarly.

### Code
[PyTorch](https://github.com/NVlabs/MUNIT)

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

## (DRIT)Diverse Image-to-Image Translation via Disentangled Representations - Lee - ECCV 2018
- Title: **Diverse Image-to-Image Translation via Disentangled Representations**
- Author: H.-Y. Lee, H.-Y. Tseng, J.-B. Huang, M. K. Singh, and M.-H. Yang
- Date: Aug. 2018.
- Arxiv: [1808.00948](https://arxiv.org/abs/1808.00948)
- Published: ECCV 2018

![DRIT- Lee - ECCV 2018](https://i.imgur.com/S3OklAy.jpg)

Comparisons of unsupervised I2I translation methods. Denote x and y as images in domain X and Y: (a) CycleGAN  maps x and y onto separated latent spaces. (b) UNIT assumes x and y can be mapped onto a shared latent space. (c) Our approach disentangles the latent spaces of x and y into a shared content space C and an attribute space A of each domain.

### Model overview
![DRIT- Lee - ECCV 2018](https://i.imgur.com/RBvyiE4.jpg)

(a) With the proposed content adversarial loss $L^{content}_{adv}$ and the cross-cycle consistency loss $L^{cc}_1$, we are able to learn the multimodal mapping between the domain X and Y with unpaired data. Thanks to the proposed disentangled representation, we can generate output images conditioned on either (b) random attributes or (c) a given attribute at test time.

### Additional loss functions
![DRIT- Lee - ECCV 2018](https://i.imgur.com/b41AujQ.jpg)

In addition to the cross-cycle reconstruction loss $L^{cc}_1$ and the content adversarial loss $L^{content}_{adv}$ described, we apply several additional loss functions in our training process. 

The self-reconstruction loss $L^{recon}_1$ facilitates training with self-reconstruction; the KL loss L aims to align the attribute representation with a prior Gaussian distribution; the adversarial loss $L^{domain}_{adv}$ encourages G to generate realistic images in each domain; and the latent regression loss $L^{latent}_1$ enforces the reconstruction on the latent attribute vector. Finally, the mode seeking regularization $L_{ms}$ further improves the diversity.

$$
\begin{aligned} L_{\mathrm{adv}}^{\mathrm{content}} &\left(E_{\mathcal{X}}^{c}, E_{\mathcal{Y}}^{c}, D^{c}\right)=\\ & \mathbb{E}_{x}\left[\frac{1}{2} \log D^{c}\left(E_{\mathcal{X}}^{c}(x)\right)+\frac{1}{2} \log \left(1-D^{c}\left(E_{\mathcal{X}}^{c}(x)\right)\right)\right.\\+& \mathbb{E}_{y}\left[\frac{1}{2} \log D^{c}\left(E_{\mathcal{Y}}^{c}(y)\right)+\frac{1}{2} \log \left(1-D^{c}\left(E_{\mathcal{Y}}^{c}(y)\right)\right)\right] \end{aligned}
$$

### Multi-domains I2I framework. 
We further extend the proposed disentangle representation framework to a more general multi-domain setting. Different from the class-specific encoders, generators, and discriminators used in two-domain I2I, all networks in mutli-domain are shared among all domains. Furthermore, one-hot domain codes are used as inputs and the discriminator will perform domain classification in addition to discrimination.

![DRIT- Lee - ECCV 2018](https://i.imgur.com/fuoFNyF.jpg)


### Multi-scale generator-discriminator
![DRIT- Lee - ECCV 2018](https://i.imgur.com/ZqrkHPW.jpg)

To enhance the quality of generated high-resolution images, we adopt a multi-scale generator-discriminator architecture. We generate low-resolution images from the intermediate features of the generator. An additional adversarial domain loss is applied on the low-resolution images.


### Code
- [Project Site](http://vllab.ucmerced.edu/hylee/DRIT_pp/)
- [PyTorch](https://github.com/HsinYingLee/DRIT)

## vid2vid - Ting-Chun Wang - NIPS 2018
- Title: **Video-to-Video Synthesis**
- Author: Ting-Chun, Wang Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, Bryan Catanzaro
- Date: Aug. 2018.
- Arxiv: [1808.06601](https://arxiv.org/abs/1808.06601)
- Published: NIPS 2018

![vid2vid - Ting-Chun Wang - NIPS 2018](https://i.imgur.com/8b5S8vG.jpg)


The network architecture (G1) for low-res videos. Our network takes in a number of semantic label maps and previously generated images, and outputs the intermediate frame as well as the flow map and the mask.

![vid2vid - Ting-Chun Wang - NIPS 2018](https://i.imgur.com/kOnSJLu.jpg)

The network architecture (G2) for higher resolution videos. The label maps and previous frames are downsampled and fed into the low-res network G1. Then, the features from the high-res network and the last layer of the low-res network are summed and fed into another series of residual blocks to output the final images.

### Code
- [Project Site](https://tcwang0509.github.io/vid2vid/)
- [PyTorch](https://github.com/NVIDIA/vid2vid)

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

### SPADE - Park - CVPR 2019
- Title: **Semantic Image Synthesis with Spatially-Adaptive Normalization**
- Author: Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu
- Date: Mar. 2019
- Arxiv: [1903.07291](https://arxiv.org/abs/1903.07291)
- Published: CVPR 2019

### Method overview
![(SPADE)Semantic Image Synthesis with Spatially-Adaptive Normalization - Park - CVPR 2019](https://i.imgur.com/kjqvqzi.jpg)

In many common normalization techniques such as Batch Normalization (Ioffe et al., 2015), there are learned affine layers (as in PyTorch and TensorFlow) that are applied after the actual normalization step. In SPADE, the affine layer is learned from semantic segmentation map. This is similar to Conditional Normalization (De Vries et al., 2017 and Dumoulin et al., 2016), except that the learned affine parameters now need to be spatially-adaptive, which means we will use different scaling and bias for each semantic label. Using this simple method, semantic signal can act on all layer outputs, unaffected by the normalization process which may lose such information. Moreover, because the semantic information is provided via SPADE layers, random latent vector may be used as input to the network, which can be used to manipulate the style of the generated images.

![SPADE - Park - CVPR 2019](https://i.imgur.com/EjyGex7.jpg)


In the SPADE generator, each normalization layer uses the segmentation mask to modulate the layer activations. (left) Structure of one residual block with SPADE. (right) The generator contains a series of SPADE residual blocks with upsampling layers. Our architecture achieves better performance with a smaller number of parameters by removing the downsampling layers of leading image-to-image translation networks 

### Detail design

![SPADE - Park - CVPR 2019](https://i.imgur.com/FTU0q3P.jpg)
In SPADE, the mask is first projected onto an embedding space, and then convolved to produce the modulation parameters $γ$ and $β$. Unlike prior conditional normalization methods, $γ$ and $β$ are not vectors, but tensors with spatial dimensions. The produced $γ$ and $β$ are multiplied and added to the normalized activation element-wise.

![SPADE - Park - CVPR 2019](https://i.imgur.com/D5Pu9v4.jpg)

The term 3x3-Conv-k denotes a 3-by-3 convolutional layer with k convolutional filters. The segmentation map is resized to match the resolution of the corresponding feature map using nearest-neighbor down-sampling.

![SPADE - Park - CVPR 2019](https://i.imgur.com/K6jsWx9.jpg)

We note that for the case that the number of channels before and after the residual block is different, the skip connection is also learned (dashed box in the figure).

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

### Code
- [Project Site](https://nvlabs.github.io/SPADE/)
- [PyTorch](https://github.com/nvlabs/spade/)
- [Online Demo](https://www.nvidia.com/en-us/research/ai-playground/)

### (INIT)Towards Instance-level Image-to-Image Translation - Shen - CVPR 2019
- Title: **Towards Instance-level Image-to-Image Translation**
- Author: Zhiqiang Shen, Mingyang Huang,  Jianping Shi, Xiangyang Xue, Thomas Huang
- Date:  May 2019
- Arxiv: [1905.01744](https://arxiv.org/abs/1905.01744)
- Published: CVPR 2019

- The instance-level objective loss can help learn a more accurate reconstruction and incorporate diverse attributes of objects
- A more reasonable mapping: the styles used for target domain of local/global areas are from corresponding spatial regions in source domain.
- A large-scale, multimodal, highly varied Image-to-Image translation dataset, containing ∼155k streetscape images across four domains. 



Disentangle background and object style in translation process:
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/AH9uHln.png)

The framework overview:
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/cMSETzP.png)

### Loss Design
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/jps66rW.png)



![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/Ui9wDkn.png)


The instance-level translation dataset and comparisons with previous ones:
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/sgUndtd.png)

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

### Code
- [Project Site](http://zhiqiangshen.com/projects/INIT/index.html)


## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
