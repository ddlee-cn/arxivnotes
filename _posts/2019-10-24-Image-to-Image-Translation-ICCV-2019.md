---
title: "ICCV 2019: Image-to-Image Translation"
tag:
- Image-to-Image Translation
- Review
---

## Guided Image-to-Image Translation With Bi-Directional Feature Transformation

In this work, the authors propose to apply the conditioning operation in both direction with information flowing not only from the guidance image to the input image, but from the input image to the guidance image as well. Second, they extend the existing feature-wise feature transformation to be spatially varying to adapt to different contents in the input image.



### Comparison of Conditioning Schemes



![Guided Image-to-Image Translation With Bi-Directional Feature Transformation](https://i.imgur.com/mglT7II.png)



There are many schemes to incorporate the additional guidance into the image-to-image translation model. One straight forward scheme is (a) input concatenation, this will assume that we need the guidance image at the first stage of the model. Another scheme is (b) feature concatenation. It assumes that we need the feature representation of the guide before upsampling.

In (c) we replace every normalization layer with our novel feature transformation (FT) layer that manipulates the input using scaling and shifting parameters generated from the guide using a parameter generator (PG). We denote this uni-directional scheme as uFT. In this work, we propose (d) a bi-directional feature transformation scheme denoted as bFT. In bFT, the input is manipulated using scaling and shifting parameters generated from the guide and the guide is also manipulated using scaling and shifting parameters generated from the input.



### Bi-directional Feature Transformation



![Guided Image-to-Image Translation With Bi-Directional Feature Transformation](https://i.imgur.com/RhyNBke.png)



We present a bi-directional feature transformation model to better utilize the additional guidance for guided image-to-image translation problems. In place of every normalization layer in the encoder, we add our novel FT layer. This layer scales and shifts the normalized feature of that layer as shown in Figure 4. The scaling and shifting parameters are generated using a parameter generation model of two convolution layers with a bottleneck of 100 dimension.


$$
F_{\text {input }}^{l+1}=\gamma_{\text {guide }}^{l} \frac{F_{\text {input }}^{l}-\operatorname{mean}\left(F_{\text {input }}^{l}\right)}{\operatorname{std}\left(F_{\text {input }}^{l}\right)}+\beta_{\text {guide }}^{l}
$$




### Spatially Varing Manner



![Guided Image-to-Image Translation With Bi-Directional Feature Transformation](https://i.imgur.com/HbJLKV3.png)



A key difference between a FiLM layer and our FT layer is that the scaling γ and shifting β parame-
ters of the FiLM layer are vectors, while in our FT layer they are tensors. Therefore, the scaling and shifting operations are applied in spatially varying manner in our FT layer in contrast to spatially invariant modulation as in the FiLM layer.






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





## RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes



### Relative Attributes

Top: Comparing facial attribute transfer via relative and target attributes. (a) Existing target-attribute-based methods do not know whether each attribute is required to change or not, thus could over-emphasize some attributes. In this example, StarGAN changes the hair color but strengthens the degree of smile. (b)
RelGAN only modifies the hair color and preserves the other attributes (including smile) because their relative attributes are zero.



![RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes](https://i.imgur.com/UXmo6fP.png)



Bottom: By adjusting the relative attributes in a continuous manner, RelGAN provides a realistic interpolation between before and after attribute transfer.



### One Generator, Three Discriminators



![RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes](https://i.imgur.com/oRyy8h0.png)



Our model consists of a single generator G and three discriminators $D = {D_{Real} , D_{Match}, D_{Interp} }$. G conditions on an input image and relative attributes (top left), and performs facial attribute transfer or interpolation (top right). During training, G aims to fool the following three discriminators (bottom): $D_{Real}$ tries to distinguish between real images and generated images. $D_{Match}$ aims to distinguish between real triplets and generated/wrong triplets. $D_{Interp}$ tries to predict the degree of interpolation.



### Conditional Adversarial Loss


$$
\begin{aligned} \min _{G} \max _{D_{\text {Match }}} \mathcal{L}_{\text {Match }} &=\mathbb{E}_{\mathbf{x}, \mathbf{v}, \mathbf{x}^{\prime}}\left[\log D_{\text {Match }}\left(\mathbf{x}, \mathbf{v}, \mathbf{x}^{\prime}\right)\right] \\ &+\mathbb{E}_{\mathbf{x}, \mathbf{v}}\left[\log \left(1-D_{\text {Match }}(\mathbf{x}, \mathbf{v}, G(\mathbf{x}, \mathbf{v}))\right)\right] \end{aligned}
$$


From this equation, we can see that $D_{Match}$ takes a triplet as input. In particular, $D_{Match}$ aims to distinguish between two types of triplets: real triplets $(x, v, x′ )$ and fake triplets $(x, v, G(x, v))$. A real triplet$ (x, v, x′)$ is comprised of two real images $(x, x′ )$ and the relative attribute vector $v = a′ − a$, where a′ and a are the attribute vector of $x′$ and $x$ respectively. 

We also propose to incorporate a third type of triplets: wrong triplet, which consists of two real images with mismatched relative attributes. By adding wrong triplets, $D_{Match}$ tries to classify the real triplets as +1 (real and matched) while both the fake and the wrong triplets as −1 (fake or mismatched). In particular, we create wrong triplets using the following simple procedure: given a real triplet expressed by $(x, a′ − a, x′)$,
we replace one of these four variables by a new one to create a wrong triplet. By doing so, we obtain four different wrong triplets.



![RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes](https://i.imgur.com/CL2xAON.png)



### Interpolation Loss

The goal of $D_{Interp}$ is to take an generated image as input and predict its degree of interpolation α̂, which is defined as $α̂ = min(α, 1 − α)$, where $α̂ = 0$ means no interpolation and $α̂ = 0.5$ means maximum interpolation. By predicting α̂, we resolve the ambiguity between α and 1 − α.
$$
\begin{aligned} \min _{D_{\text {lief }}} \mathcal{L}_{\text {literp }}^{D}=\mathbb{E}_{\mathbf{x}, \mathbf{v}, \alpha}\left[\left\|D_{\text {luterp }}(G(\mathbf{x}, \alpha \mathbf{v}))-\hat{\alpha}\right\|^{2}\right.\\+\left\|D_{\text {litherp }}(G(\mathbf{x}, \mathbf{0}))\right\|^{2} \\ &\left.+\left\|D_{\text {Interp }}(G(\mathbf{x}, \mathbf{v}))\right\|^{2}\right] \end{aligned}
$$


where the first term aims at recovering â from $G(x, αv)​$. The second and the third term encourage $D_{Interp}$ to output zero for the non-interpolated images. The objective function of G is modified by adding the following loss:
$$
\begin{aligned} \min _{D_{\text {litarp }}} \mathcal{L}_{\text {lnterp }}^{D}=\mathbb{E}_{\mathbf{x}, \mathbf{v}, \alpha} &\left[\left\|D_{\text {Interp }}(G(\mathbf{x}, \alpha \mathbf{v}))-\hat{\alpha}\right\|^{2}\right.\\+&\left.\left\|D_{\text {Interp }}(G(\mathbf{x}, \mathbb{I}[\alpha>0.5] \mathbf{v}))\right\|^{2}\right] \end{aligned}
$$


where $I[·]$ is the indicator function that equals to 1 if its argument is true and 0 otherwise. 



![RelGAN: Multi-Domain Image-to-Image Translation via Relative Attributes](https://i.imgur.com/vZRtf7k.png)






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





## Learning Fixed Points in Generative Adversarial Networks: From Image-to-Image Translation to Disease Detection and Localization

Can GANs remove an object, if present, from an image while otherwise preserving the image content? 

### Ideal Requirements

- Req. 1: The GAN must handle unpaired images. It may be too arduous to collect a perfect pair of photos of the same person with and without eyeglasses, and it would be too late to acquire a healthy image for a patient with an illness undergoing medical imaging.
- Req. 2: The GAN must require no source domain label when translating an image into a target domain (i.e., source-domain-independent translation). For instance, a GAN trained for virtual healing aims to turn any image, with unknown health status, into a healthy one.
- Req. 3: The GAN must conduct an identity transformation for same-domain translation. For “virtual healing”, the GAN should leave a healthy image intact, injecting neither artifacts nor new information into the image.
- Req. 4: The GAN must perform a minimal image transformation for cross-domain translation. Changes should be applied only to the image attributes directly relevant to the translation task, with no impact on unrelated attributes. For instance, removing eyeglasses should not affect the remainder of the image (e.g., the hair, face color, and background), or removing diseases from a diseased image should not impact the region of the image labeled as normal.

### Fixed-Point GAN Training Scheme



![Learning Fixed Points in Generative Adversarial Networks: From Image-to-Image Translation to Disease Detection and Localization](https://i.imgur.com/ST5SMoj.png)



Similar to StarGAN, our discriminator learns to distinguish real/fake images and classify the domains of input images (1A–B). However, unlike StarGAN, our generator learns to perform not only cross-domain translations via transformation learning (2A–B), but also same-domain translations via fixed-point translation learning (3A–C), which is essential for mitigating the limitations of StarGAN and realizing disease detection and localization using only image-level annotation.



### Conditional Identity Loss

During training, StarGAN focuses on translating the input image to different target domains. This strategy cannot penalize the generator when it changes aspects of the input that are irrelevant to match target domains. In addition to learning a translation to different domains, we force the generator, using the con-
ditional identity loss, to preserve the domain identity while translating the image to the source domain. This
also helps the generator learn a minimal transformation for translating the input image to the target domain.


$$
\mathcal{L}_{i d}=\left\{\begin{array}{ll}{0,} & {c=c_{y}} \\ {\mathbb{E}_{x, c}\left[| G(x, c)-x \|_{1}\right],} & {c=c_{x}}\end{array}\right.
$$






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




## Deep CG2Real: Synthetic-to-Real Translation via Image Disentanglement

We leverage the intrinsic images disentanglement to change this problem from a purely unpaired setting to a two-stage paired-unpaired setting. We render the synthetic scenes with a physically-based renderer (“PBR”) to simulate realistic illumination and create paired OpenGL-PBR shading data. In the first stage of our pipeline, we use this data to train an OpenGL-to-PBR shading translation network that synthesizes realistic shading. We combine this new shading with the original OpenGL textures to reconstruct our intermediate PBR images.



![](https://i.imgur.com/a4TULcK.png)



In the second stage, we translate these PBR images to the real image domain in an unsupervised manner using a CycleGAN-like network. We train individual PBR-to-real generators for the shading and albedo layers; we use an encoder-decoder architecture for shading to increase global context, and a purely convolutional network, with no downsampling/unsampling for albedo. As in CycleGAN, the quality of this translation is best with a backward real-to-PBR cycle, and a cycle-consistency loss. In the absence of the disentangled shading-albedo layers for real images, we accomplish this with an asymmetric architecture and a PBR-domain intrinsic image decomposition network.



![Deep CG2Real: Synthetic-to-Real Translation via Image Disentanglement](https://i.imgur.com/2HaRgUk.png)



To finish the backward cycle, the real image is first translated to the PBR domain. Afterwards we use the pretrained intrinsic decomposition network H to decompose it into its albedo and shading, which are further fed to corresponding generators. Finally we multiply the output albedo and shading to reconstruct the original real image.



## Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation

### 2-Stage Manner

The model first completes the user input and then generates an image conditioned on the completed shape. There are several advantages to  this two-stage approach. For one, we are able to give the
artist feedback on the general object shape in our interactive interface , allowing them to quickly refine higher level shape until it is satisfactory. Second, we found that splitting completion and image generation to work better than going directly from partial outlines to images, as the additional intermediate supervision on full outlines/sketches breaks the problem into two easier sub-problems – first recover the geometric properties of the object (shape, proportions) and then fill in the appearance(colors, textures).




![Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation](https://arnabgho.github.io/iSketchNFill/resources/imgs/method.png)

First, complete a partial sketch using the shape generator $G_S$ . Then translate the completed sketch into an image using the appearance generator $G_A$ . Both generators are trained with their respective discriminators $D_S$ , and $D_A$.





### Stage 1: Sketch Completion

To achieve multi-modal completions, the shape generator is designed using inspiration from non-image conditional model with the conditioning input provided at multiple scales, so that the generator network doesn't ignore the partial stroke conditioning.



![Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation](https://i.imgur.com/9gEjSrP.png)



### Stage 2: Sketch-to-Image Translation

For the second stage, we use a multi-class generator that is conditioned on a user supplied class label. This generator applies a gating mechanism that allows the network to focus on the important parts (activations) of the network specific to a given class. Such an approach allows for a clean separation of classes, enabling us to train a single generator and discriminator across multiple object classes, therebyenabling a finite-size deployable model that can be used in multiple different scenarios.


### Gating Mechanism



![Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation](https://i.imgur.com/FJ2P4KO.png)



The model uses gating on all the residual blocks of the generator and the discriminator, other forms of conditioning such as (naive concatenation in input only, all layers, AC-GAN like latent regressor are evaluated as well.



![Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation](https://i.imgur.com/PZIFGrI.png)



(Left) A “vanilla” residual block without conditioning applies a residual modification to the input tensor. 

(Mid-left) The $H(X)$ block is softly-gated by scalar parameter α and shift $β$. 

(Mid) Adaptive Instance Normalization applies a channel-wise scaling and shifting after an instance normalization layer. 

(Mid-right) Channel-wise gating adds restrictions Class to label the range injection of $α$. 

(Right) We find that channel-wise gating (without added bias) produces the best results empirically.






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




## (FUNIT)Few-Shot Unsupervised Image-to-Image Translation


![(FUNIT)Few-Shot Unsupervised Image-to-Image Translation](https://nvlabs.github.io/FUNIT/images/problem.jpg)



While unsupervised/unpaired image-to-image translation methods (e.g., [Liu and Tuzel](https://arxiv.org/abs/1606.07536), [Liu et. al.](https://arxiv.org/abs/1703.00848), [Zhu et. al.](https://arxiv.org/abs/1703.10593), and [Huang et. al.](https://arxiv.org/abs/1804.04732)) have achieved remarkable success, they are still limited in two aspects.

First, they generally require seeing a lot of images from target class in the training time.

Second, a trained model for a translation task cannot be repurposed for another translation task in the test time.

We propose a few-shot unsupervised image-to-image translation framework (FUNIT) to address the limitation. In the training time, the FUNIT model learns to translate images between any two classes sampled from a set of source classes. In the test time, the model is presented a few images of a target class that the model has never seen before. The model leverages these few example images to translate an input image of a source class to the target class.

### The Generator

To generate a translation output $x̄$, the translator combines the class latent code z y extracted from the class images $y_1$,..., $y_k$ with the content latent code z x extracted from the input content image. Note that nonlinearity and normalization operations are not included in the visualization.



![(FUNIT)Few-Shot Unsupervised Image-to-Image Translation](https://i.imgur.com/i3gWfbQ.png)




### The Multi-task Discriminator

Our discriminator D is trained by solving multiple adversarial classification tasks simultaneously. Each of the tasks is a binary classification task determining whether an input image is a real image of the source class or a translation output coming from G. As there are $S$ source classes, D produces $S$ outputs. When updating D for a real image of source class $c_X$, we penalize D if its $c_X$th output is false. For a translation output yielding a fake image of source class $c_X$, we penalize D if its $c_X$th output is positive. We do not penalize D for not predicting false for images of other classes (S\{$c_X$}). When updating G, we only penalize G if the $c_X$ th output of D is false.



## Related

- [Few-shot Video-to-Video Synthesis](https://arxivnote.ddlee.cn/2019/10/28/Few-Shot-Video-to-Video-Synthesis-NIPS.html)
- [Few-Shot Unsupervised Image-to-Image Translation](https://arxivnote.ddlee.cn/2019/10/27/Few-Shot-Unsupervised-Image-to-Image-Translation-ICCV.html)
- [Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation - ICCV 2019 - PyTorch](https://arxivnote.ddlee.cn/2019/10/12/Interactive-Sketch-Fill-Multiclass-Sketch-Image-Translation-ICCV-PyTorch.html)
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/2019/08/21/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE and INIT](https://arxivnote.ddlee.cn/2019/08/22/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [(DMIT)Multi-mapping Image-to-Image Translation via Learning Disentanglement - Xiaoming Yu - NIPS 2019](https://arxivnote.ddlee.cn/2019/10/08/Multi-mappitng-Image-to-Image-Translation-Disentanglement.html)
- [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation - Junho Kim - 2019](https://arxivnote.ddlee.cn/2019/09/18/U-GAT-IT-Unsupervised-Generative-Attentional-Networks-with-Adaptive-Layer-Instance-Normalization-for-Image-to-Image-Translation-Junho-Kim-2019.html)
- [Towards Instance-level Image-to-Image Translation - Shen - CVPR 2019](https://arxivnote.ddlee.cn/2019/07/18/Towards-Instance-level-Image-to-Image-Translation-Shen-CVPR-2019.html)

