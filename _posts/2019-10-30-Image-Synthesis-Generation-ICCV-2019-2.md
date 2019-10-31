---
title: "ICCV 2019: Image Synthesis(Part 2)"
tag:
- Image Synthesis
- Review
---





## SinGAN: Learning a Generative Model From a Single Natural Image(Best Paper Award)



[Project Page](http://webee.technion.ac.il/people/tomermic/SinGAN/SinGAN.htm) | [Code](<https://github.com/tamarott/SinGAN>)




### Multi-Scale Pipeline

![SinGAN: Learning a Generative Model From a Single Natural Image](https://i.imgur.com/C9SoOvE.png)

Our model consists of a pyramid of GANs, where both training and inference are done in a coarse-to-fine fashion. At each scale, Gn learns to generate image samples in which all the overlapping patches cannot be distinguished from the patches in the down-sampled training image, xn , by the discriminator Dn ; the effective patch size decreases as we go up the pyramid (marked in yellow on the original image for illustration). The input to Gn is a random noise image zn , and the generated image from the previous scale x̃n, upsampled to the current resolution (except for the coarsest level which is purely generative). The generation process at level n involves all generators {GN . . . Gn } and all noise maps {zN , . . . , zn } up to this level.

$$
\tilde{x}_{n}=\left(\tilde{x}_{n+1}\right) \uparrow^{r}+\psi_{n}\left(z_{n}+\left(\tilde{x}_{n+1}\right) \uparrow^{r}\right)
$$


### Single Scale Generation

![SinGAN: Learning a Generative Model From a Single Natural Image](https://i.imgur.com/XeDZZMx.png)

At each scale n, the image from the previous scale, $x̃_{n+1}$, is upsampled and added to the input noise map, zn . The result is fed into 5 conv layers, whose output is a residual image that is added back to $\left(\tilde{x}_{n+1}\right) \uparrow^{r}$ . This is the output $x̃n$ of Gn.





### Loss Functions

$$
\min _{G_{n}} \max _{D_{n}} \mathcal{L}_{\mathrm{adv}}\left(G_{n}, D_{n}\right)+\alpha \mathcal{L}_{\mathrm{rec}}\left(G_{n}\right)
$$

Adversarial Loss and Reconstruction Loss

$$
\mathcal{L}_{\mathrm{rec}}=\left\|G_{n}\left(0,\left(\tilde{x}_{n+1}^{\mathrm{rec}}\right) \uparrow^{r}\right)-x_{n}\right\|^{2}
$$



Full notes with code: [SinGAN: Learning a Generative Model From a Single Natural Image - ICCV 2019 - PyTorch](https://arxivnote.ddlee.cn/2019/10/23/SinGAN-Generative-Model-Single-Natural-Image.html)



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




## InGAN: Capturing and Retargeting the “DNA” of a Natural Image(Oral)



[Project Page](<http://www.wisdom.weizmann.ac.il/~vision/ingan/>) | [Code](<https://github.com/assafshocher/InGAN>)



### Overview

![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](https://i.imgur.com/BhyGuLC.png)

InGAN consists of a Generator G that retargets input x to output y whose size/shape is determined by a geometric transformation T (top left). A multiscale discriminator D learns to discriminate the patch statistics of the fake output y from the true patch statistics of the input image (right). Additionally, we take advantage of G’s automorphism to reconstruct the input back from y using G and the inverse transformation T −1 (bottom left).

The formulation aims to achieve two properties:

1. matching distributions: The distribution of patches, across scales, in the synthesized image, should match that distribution in the original input image. This property is a generalization of both the Coherence and Completeness objectives. 
2. localization: The elements’ locations in the generated image should generally match their relative locations in the original input image.



### Shape-flexible Generator

![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](https://i.imgur.com/FrsxWLG.png)

The desired geometric transformation for the output shape T is treated as an additional input that is fed to G for every forward pass. A parameter-free transformation layer geometrically transforms the feature map to the desired output shape. Making the transformation layer
parameter-free allows training G once to transform x to any size, shape or aspect ratio at test time.

### Multi-scale Patch Discriminator

![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](https://i.imgur.com/YgWSLpD.png)

InGAN uses a multi-scale D. This feature is significant: A single scale discriminator can only capture patch statistics of a specific size. Using a multiscale D matches the patch distribution over a range of patch sizes, capturing both fine-grained details as well as coarse structures in the image. At each scale, the discriminator is rather simple: it consists of just four conv-layers with the first one strided. Weights are not shared between different scale discriminators.




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




## Specifying Object Attributes and Relations in Interactive Scene Generation(Best Paper Honorable Mentions)



![Specifying Object Attributes and Relations in Interactive Scene Generation](https://i.imgur.com/5OCPTD4.png)



(top row) the schematic illustration panel of the user interface, in which the user arranges the desired objects. (2nd row) the scene graph that is inferred automatically based on this layout. (3rd row) the layout that is created from the scene graph. (bottom row) the generated image. Legend for the GUI colors in the top row: purple – adding an object, green – resizing it, red – replacing its appearance. (a) A simple layout with a sky object, a tree and a grass object. All object appearances are initialized to a random archetype appearance. (b) A giraffe is added. (c) The giraffe is enlarged. (d) The appearance of the sky is changed to a different archetype. (e) A small sheep is added. (f) An airplane is added. (g) The tree is enlarged.



![Specifying Object Attributes and Relations in Interactive Scene Generation](https://i.imgur.com/hmXHK1I.png)



The architecture of our composite network, including the subnetworks G, M, B, A, R, and the process of creating the layout tensor t. The scene graph is passed to the network G to create the layout embedding ui of each object. The bounding box bi is created from this embedding, using network B. A random vector zi is concatenated to ui , and the network M computes the mask mi . The appearance information, as encoded by the network A, is then added to create the tensor t with c + d5 channels, c being the number of classes. The autoencoder R generates the final image p from this tensor.



Full notes with code: [Specifying Object Attributes and Relations in Interactive Scene Generation](https://arxivnote.ddlee.cn/2019/10/20/Specifying-Object-Attributes-Relations-Interactive-Scene-Generation-ICCV-2019.html)




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




## Content and Style Disentanglement for Artistic Style Transfer(Oral)

[Project Page](https://compvis.github.io/content-style-disentangled-ST/)

### Overview



![Content and Style Disentanglement for Artistic Style Transfer](https://i.imgur.com/tx8Gvpn.png)



The training iteration is performed on a pair of content images with content representations c1, c2 and a pair of style images having style representations s1 , s2 . In a next step, image pairs are fed into the content encoder Ec and style encoder Es respectively. Now we generate all possible pairs of content and style representations using the decoder D. The resulting images are fed into the style encoder Es one more time to compute the $L_{FPT−style}$ on two triplets that share c1/s1 by comparing the style representations of generated images with the styles c1 /s1, c2 /s1, c1 /s2, c2 /s2 to the styles s1 , s2 of the input style images. The resulting images are given to the discriminator D to compute a conditional adversarial loss $L_{adv}$ and to Ec to compute the discrepancy $L_{FP−content}​$ between the stylizations c2 /s2 , c1 /c1 and the original c1 , c2 . Both depicted encoders Es are shared as well as both encoders Ec.



### Fixpoint Triplet Loss


$$
\begin{array}{c}{\mathcal{L}_{F P T-s t y l e}:=\sum_{x \sim \mathbf{X}} \mathbb{E}_{\mathbf{x}} \max (0} \\ {\left(y_{1}, s_{1}\right),\left(y_{2}, s_{2}\right) \sim \mathbf{Y}} \\ {r+\left\|E_{s}\left(y_{1}\right)-E_{s}\left(D\left(E_{c}(x), E_{s}\left(y_{1}\right)\right)\right)\right\|^{2}-} \\ {\left.\left\|E_{s}\left(y_{1}\right)-E_{s}\left(D\left(E_{c}(x), E_{s}\left(y_{2}\right)\right)\right)\right\|^{2}\right)}\end{array}
$$




### Disentanglement Loss

The $L_{FPD}$ penalizes the model for perturbations which are too large in the style representation: if given the style vector $s = E_s (y)$, then the style discrepancy of two stylizations is larger than the discrepancy between stylization and original style.

$$
\begin{array}{c}{\mathcal{L}_{F P D}=\underset{x_{1}, x_{2} \sim \mathbf{X} \atop(y, s) \sim \mathbf{Y}}{\mathbb{E}} \max (0,} \\ \left\|E_{s}\left(D\left(E_{c}\left(x_{1}\right), E_{s}(y)\right)\right)-E_{s}\left(D\left(E_{c}\left(x_{2}\right), E_{s}(y)\right)\right)\right\|^{2}- \\ {\left.\left\|E_{s}\left(D\left(E_{c}\left(x_{1}\right), E_{s}(y)\right)\right)-E_{s}(y)\right\|^{2}\right)}\end{array}
$$





## Lifelong GAN: Continual Learning for Conditional Image Generation



![Lifelong GAN: Continual Learning for Conditional Image Generation](https://i.imgur.com/9zB7B0d.png)



Traditional training methods suffer from catastrophic forgetting: when we add new tasks, the network forgets how to perform previous tasks. The proposed Lifelong GAN is a generic framework for conditional image generation that applies to various types of conditional inputs (e.g. labels and images).



![Lifelong GAN: Continual Learning for Conditional Image Generation](https://i.imgur.com/DbYm9m6.png)

(Based on BicycleGAN)



Given training data for the $t^{th}$ task, model Mt is trained to learn this current task. To avoid forgetting previous tasks, knowledge distillation is adopted to distill information from model Mt−1 to model Mt by encouraging the two networks to produce similar output values or patterns given the auxiliary data as inputs.



## Related

- [ICCV 2019: Image Synthesis(Part One)](https://arxivnote.ddlee.cn/2019/10/30/Image-Synthesis-Generation-ICCV-2019.html)
- [ICCV 2019: Image Synthesis(Part Two)](https://arxivnote.ddlee.cn/2019/10/30/Image-Synthesis-Generation-ICCV-2019-2.html)
- [ICCV 2019: Image and Video Inpainting](https://arxivnote.ddlee.cn/2019/10/26/Image-Video-Inpainting-ICCV-2019.html)
- [ICCV 2019: Image-to-Image Translation](https://arxivnote.ddlee.cn/2019/10/24/Image-to-Image-Translation-ICCV-2019.html)
- [ICCV 2019: Face Editing and Manipulation](https://arxivnote.ddlee.cn/2019/10/29/Face-Editing-Manipulation-ICCV-2019.html)
- [GANs for Image Generation: ProGAN, SAGAN, BigGAN, StyleGAN](https://cvnote.ddlee.cn/2019/09/15/ProGAN-SAGAN-BigGAN-StyleGAN.html)
- [Deep Generative Models(Part 3): GANs(from GAN to BigGAN)](https://arxivnote.ddlee.cn/2019/08/20/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/2019/08/19/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/2019/08/18/Deep-Generative-Models-Taxonomy-VAE.html)
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/2019/08/21/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE and INIT](https://arxivnote.ddlee.cn/2019/08/22/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)

