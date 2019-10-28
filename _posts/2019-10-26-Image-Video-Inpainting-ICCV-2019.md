---
title: "ICCV 2019: Image and Video Inpainting"
tag:
- Image Inpainting
- Video Inpainting
- Content Generation
---

## StructureFlow: Image Inpainting via Structure-Aware Appearance Flow

Structure(edge-preserved smoothing) Generator + Apperace Flow for Texture Generation


Overview of our StructureFlow. Our model first generates global structures (i.e. edge-preserved smooth images) using structure reconstructor. Then texture generator is used to yield high-frequency details and output the final results. We add the appearance flow to our texture generator to sample features from existing regions.

The edge-preserved smooth methods aim to remove high-frequency textures while retaining the sharp edges and low-frequency structures. Their results can well represent global structures.


![StructureFlow: Image Inpainting via Structure-Aware Appearance Flow](https://i.imgur.com/wyMF61D.png)


It is hard for convolutional neural networks to model long-term correlations. In order to establish a clear relationship between different regions, we propose to use appearance flow to sample features from regions with similar structures. Since appearance flow is easily stuck within bad local minima in the inpainting task, in this work, we made two modifications to ensure the convergence of the training process.First, Gaussian sampling is employed instead of Bilinear sampling to expand the receptive field of the sampling operation. Second, we introduce a new loss function, called sampling correctness loss, to determine if the correct regions are sampled.


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



## Coherent Semantic Attention for Image Inpainting

The authors divide the image inpainting into two steps. The first step is constructed by training a rough network to rough out the missing contents. A refinement network with the CSA layer in encoder guides the second step to refine the rough predictions. In order to make network training process more stable and motivate the CSA layer to work better, we propose a consistency loss to measure not only the distance between the VGG feature layer and the CSA layer but also the distance between the VGG feature layer and the the corresponding layer of the CSA in decoder. Meanwhile, in addition to a patch discriminator, we improve the details by introducing a feature patch discriminator which is simpler in formulation, faster and more stable for training than conventional one. Except for the consistency loss, reconstruction loss and relativistic average LS adversarial loss are incorporated as constraints to instruct our model to learn meaningful parameters

![Coherent Semantic Attention for Image Inpainting](https://i.imgur.com/0oqZbbO.png)


### CSA Layer

In the painting process, one always continues to draw new lines and coloring from the end nodes of the lines drawn previously, which actually ensures the local pixel continuity of the final result.

Inspired by this process, we propose a coherent semantic attention layer (CSA), which fills in the unknown regions of the image feature maps with the similar process. Initially, each unknown feature patch in the unknown region is initialized with the most similar feature patch in the known regions. Thereafter, they are iteratively optimized by considering the spatial consistency with adjacent patches. Consequently, the global semantic consistency is guaranteed by the first step, and the local feature coherency is maintained by the optimizing step.


![Coherent Semantic Attention for Image Inpainting](https://i.imgur.com/jZF8l3w.png)

Firstly, we search the most similar contextual patch mi of each generated patch mi in the hole M , and initialize mi with mi . Then, the previous generated patches and the most similar contextual patch are combined to generate the current one.

$$
\begin{array}{c}{m_{1}=\overline{m_{1}}, \operatorname{Dad}_{1}=0} \\ {\operatorname{Dad}_{i}} \\ {m_{i}=\frac{\operatorname{Dad}_{i}}{\operatorname{Dad}_{i}+\operatorname{Dmax}_{i}} \times m_{(i-1)}+} \\ {\frac{\operatorname{Dmax}_{i}}{\operatorname{Dad}_{i}+\operatorname{Dmax}_{i}} \times \overline{m_{i}}}\end{array}
$$

![Coherent Semantic Attention for Image Inpainting](https://i.imgur.com/sjcgJFd.png)

The consistency loss is defined as

$$
\begin{aligned} L_{c}=& \sum_{y \in M}\left\|C S A\left(I_{i p}\right)_{y}-\Phi_{n}\left(I_{g t}\right)_{y}\right\|_{2}^{2}+\\ &\left\|C S A_{d}\left(I_{i p}\right)_{y}-\Phi_{n}\left(I_{g t}\right)_{y}\right\|_{2}^{2} \end{aligned}
$$

Where $\phi_n$ is the activation map of the selected layer in VGG-16. CSA(.) denotes the feature after the CSA layer and CSAd(.) is the corresponding feature in the decoder.


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

[PyTorch](https://github.com/KumapowerLIU/CSA-inpainting)


## Image Inpainting With Learnable Bidirectional Attention Maps

In this paper, we present a learnable attention map module for learning feature re-normalization and mask-updating in an end-to-end manner, which is effective in adapting to irregular holes and propagation of convolution layers. Furthermore, learnable reverse attention maps are introduced to allow the decoder of U-Net to concentrate on filling in irregular holes instead of reconstructing both holes and known regions, resulting in our learnable bidirectional attention maps.


### Overal Network

![Image Inpainting With Learnable Bidirectional Attention Maps](https://i.imgur.com/kgwVYLK.png)


### Learnable Attention Maps


Interplay models between mask and intermediate feature for PConv and our learnable bidirectional attention maps. Here, the white holes in Min denotes missing region with value 0, and the black area denotes the known region with value 1.


![Image Inpainting With Learnable Bidirectional Attention Maps](https://i.imgur.com/Mc7TCMe.png)


First, to make the mask adaptive to irregular holes and propagation along with layers, we substitut  ${k}_{1/9}$ with layer-wise and learnable convolution filters $k_M$ .

Second, instead of hard 0-1 mask-updating, we modify the  activation function for updated mask as,

$$
g_{M}\left(\mathbf{M}^{c}\right)=\left(\operatorname{Re} L U\left(\mathbf{M}^{c}\right)\right)^{\alpha}
$$

Third, we introduce an asymmetric Gaussian-shaped form as the activation function for attention map, 

$$
g_{A}\left(\mathbf{M}^{c}\right)=\left\{\begin{array}{ll}{a \exp \left(-\gamma_{l}\left(\mathbf{M}^{c}-\mu\right)^{2}\right),} & {\text { if } \mathbf{M}^{c}<\mu} \\ {1+(a-1) \exp \left(-\gamma_{r}\left(\mathbf{M}^{c}-\mu\right)^{2}\right),} & {\text { else }}\end{array}\right.
$$

where a, μ, γl , and γr are the learnable parameters, we initialize them as a = 1.1, μ = 2.0, γl = 1.0, γr = 1.0 and learn them in an end-to-end manner.





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

[PyTorch](https://github.com/Vious/LBAM_inpainting/)






## Progressive Reconstruction of Visual Structure for Image Inpainting

- Progressive manner
- Visual Structure Generation

![Progressive Reconstruction of Visual Structure for Image Inpainting](https://i.imgur.com/nFVrH5L.png)

Progressive Reconstruction of Visual Structure. A small part of the new structure is produced in each VSR layer. At the beginning, the known information is limited and so the encoding layers only estimate the outer parts of the missing structure. As the information accumulates during the feeding forward procedure, the decoding layers can have the capability to restore the missing inner parts. The generated parts are collected and sent to discriminator simultaneously.

![Progressive Reconstruction of Visual Structure for Image Inpainting](https://i.imgur.com/QCnPKYr.png)

### Overal Architecture

The VSR Layer is put in the first two layers and last two layers in our network. The generated structure and feature maps are sent to next and decoding layers. Finally, two structure (edge) maps of different scales are generated to learn structure information.

![Progressive Reconstruction of Visual Structure for Image Inpainting](https://i.imgur.com/Aaz0Qs8.png)


### Visual Structure Reconstruction(VSR)


Structure part is generated by a partial convolution followed by a residual block, then combined with input structure.

![Progressive Reconstruction of Visual Structure for Image Inpainting](https://i.imgur.com/rqOftPc.png)

$$
\mathbf{X}_{p c 1}, \mathbf{M}_{p c 1}=P \operatorname{conv}\left(\left\langle\mathbf{X}_{i n}, \mathbf{E}_{i n}\right\rangle,\left\langle\mathbf{M}_{i n}^{I m g}, \mathbf{M}_{i n}^{E d g e}\right\rangle\right)
$$

The input structure $E_{in}$ is used to replace the previously known area in the new structure map Econv and so only the newly generated parts $E_{conv} \odot (M_{pc1} - M_{in} )$ are preserved. 

$$
\mathbf{E}_{E G}=\mathbf{E}_{c o n v} \odot\left(\mathbf{M}_{p c 1}-\mathbf{M}_{i n}^{E d g e}\right)+\mathbf{M}_{i n}^{E d g e}
$$

Progressively,
$$
\mathbf{X}_{o u t}, \mathbf{M}_{p c 2}=\mathbf{M}_{p c 1} \odot\left(P \operatorname{conv}\left(\left\langle\mathbf{X}_{i n}, \mathbf{E}_{E G}\right\rangle,\left\langle\mathbf{M}_{i n}, \mathbf{M}_{p c 1}\right\rangle\right)\right)
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




### Code

[PyTorch](https://github.com/jingyuanli001/PRVS-Image-Inpainting)


## Free-Form Image Inpainting With Gated Convolution

We propose gated convolution for free-form image inpainting. It learns a dynamic feature gating mechanism for each channel and each spatial location (for example, inside or outside masks, RGB channels or user-guidance channels). Specifically we consider the formulation where the input feature is firstly used to compute gating values g = σ(wg x) (σ is sigmoid function, wg is learnable parameter). The final output is a multiplication of learned feature and gating values y = φ(wx)⊙g where φ can be any activation function. Gated convolution is easy to implement and performs significantly better when (1) the masks have arbitrary shapes and (2) the inputs are no longer simply RGB channels with a mask but also have conditional inputs like sparse sketch. 


### Gated Convolution

$$
\begin{aligned} \text {Gating}_{y, x} &=\sum \sum W_{g} \cdot I \\ \text {Feature}_{y, x} &=\sum \sum W_{f} \cdot I \\ O_{y, x} &=\phi\left(\text {Feature}_{y, x}\right) \odot \sigma\left(\text {Gating}_{y, x}\right) \end{aligned}
$$

![Free-Form Image Inpainting With Gated Convolution](https://i.imgur.com/HKHJXPc.png)


### Overall Architecture

For coarse and refinement networks, we use a simple encoder-decoder network instead of U-Net used in PartialConv. We found that skip connections in a U-Net have no significant effect for non-narrow mask. This is mainly because for center of a masked region, the inputs of these skip connections are almost zeros thus cannot propagate detailed color or texture information to the decoder of that region. For hole boundaries, our encoder-decoder architecture equipped with gated convolution is sufficient to generate seamless results.

![Free-Form Image Inpainting With Gated Convolution](https://i.imgur.com/MkYW9Mi.png)


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

[TensorFlow](https://github.com/JiahuiYu/)

## Free-Form Video Inpainting With 3D Gated Convolution and Temporal PatchGAN

Specifically, we observe that input videos contain many masked voxels that are potentially harmful to vanilla convolutions, we design a generator with 3D gated convolutional layers that could properly handle the masked video by learning the difference between the unmasked region, filled in region and masked region in each layer and attend on proper features correspondingly. In addition, different from image inpainting, video inpainting has to be temporally coherent, so we propose a novel Temporal PatchGAN discriminator that penalizes high-frequency spatial-temporal features and enhances the temporal consistency through the combination of different losses. We also design a new algorithm to generate diverse free-form video masks, and collect a new video dataset based on existing videos that could be used to train and evaluate learning-based video inpainting models.

![ Free-Form Video Inpainting With 3D Gated Convolution and Temporal PatchGAN](https://i.imgur.com/TfxNhVh.png)

Our model is composed of (a) a video inpainting generator with 3D gated convolutional layers that fully utilizes information for neighboring frames to handle irregular video masks and (b) a Temporal PatchGAN (T-PatchGAN) discriminator that focuses on different spatial-temporal features to enhance output quality. (c) The visualization of learned gating values σ(Gatingt,x,y ). The 3D gated convolution will attend on the masked area and gradually fill in the missing feature points. Note that all gating values are extracted from the first channel of each layer without manual picking.


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


## Copy-and-Paste Networks for Deep Video Inpainting

- Propose a self-supervised deep alignment networks that can compute affine matrices between images that contain large holes.
- Propose a novel context-matching algorithm to combine reference frame features based on similarity between images.



![Copy-and-Paste Networks for Deep Video Inpainting](https://i.imgur.com/2nHVuAI.png)

The key components of our DNN system are the alignment and the context matching. To find corresponding pixels in other frames for the holes in the given frame, the frames need to be registered first. We propose a self-supervised alignment networks, which estimates affine matrices between frames. While DNNs for computing the affine matrix or homography exist, our alignment method is able to deal with holes in images when computing the affine matrices. After the alignment, the novel context matching algorithm is used to compute the similarity between the target frame and the reference frames. The network learns which pixels are valuable for copying through the context matching, and those pixels are used to paste and complete an image. By progressively updating the reference frames with the inpainted results at each step, the algorithm can produce videos with temporal consistency.

![Copy-and-Paste Networks for Deep Video Inpainting](https://i.imgur.com/IUiDyak.png)



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



