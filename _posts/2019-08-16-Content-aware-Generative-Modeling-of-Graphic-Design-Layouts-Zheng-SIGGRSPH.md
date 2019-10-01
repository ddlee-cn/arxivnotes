---
title: Content-aware Generative Modeling of Graphic Design Layouts - Zheng - SIGGRSPH 2019
tag:
- Art
- Application
- GAN
redirect_from: /Content-aware-Generative-Modeling-of-Graphic-Design-Layouts-Zheng-SIGGRSPH.html
---



## Info
- Title: **Content-aware Generative Modeling of Graphic Design Layouts**
- Task: **Layout Design**
- Author: X. Zheng, X. Qiao, Y. Cao, and R. W. H. Lau
- Date:  Jul. 2019
- Published: SIGGRSPH 2019
- Affiliation: CityU HONG KONG

## Highlights & Drawbacks
- The first content-aware deep generative model for graphic design layouts, which is able to synthesize diverse graphic design layouts based on visual and textual features.
- A large-scale magazine layout dataset with rich semantic annotations including categories, fine-grained se- mantic layouts and keywords summarizing the text contents

## Motivation & Design
**The dataset**
A corpus of 3,919 magazine pages from the Internet, covering 6 common categories, including fash- ion, food, news, science, travel and wedding. As these 6 categories of magazine pages cover a large variety of con- tents, they exhibit a rich layout variation. We annotate each page with 6 different semantic elements, including Text, Image, Headline, Text-over-image, Headline-over-image and Background. In addition, we also extract keywords from the text contents of each page to represent the text. 

![Content-aware Generative Modeling of Graphic Design Layouts - Zheng - SIGGRSPH 2019](https://i.imgur.com/pySZVNF.jpg)

**The framework of model**
It has two main parts: a multi-modal embedding network and a layout generative network. The multi-modal embedding network learns the multi-modal features y from three inputs: visual contents (images), textual contents (keywords) and 3 high-level design attributes (design category, text proportion Tp , and image proportion Ip ). These inputs are first sent to 3 independent encoders, i.e., image encoder, text encoder and attribute encoder, respectively, and then merged via a fusion module to obtain $y$. The layout generative network learns a distribution of layouts conditioned on $y$ and extracts content-aware features $\hat{z}$, In particular, a layout encoder E maps a layout sample x to features $\hat{z}$ conditioned on $y$, a layout generator $G$ maps a random vector $z$ to a layout sample $x ̃$ conditioned on $y$, and a discriminator $D$ learns to distinguish joint pairs $(x, \hat{z}) $and $(x ̃ , z)$ conditioned on $y$. 


**Loss function**
The loss of discriminator: Least square GAN loss
$$
L_{G A N}^{D}=\frac{1}{2}(D(\mathbf{x}, E(\mathbf{x}, \mathbf{y}), \mathbf{y})-1)^{2}+\frac{1}{2}(D(G(\mathbf{z}, \mathbf{y}), \mathbf{z}, \mathbf{y}))^{2}
$$
The loss of generator: Least square GAN loss, reconstruction loss and Kullback-Leibler (KL) divergence loss 
$$
L_{G A N}^{G}=\frac{1}{2}(D(G(\mathbf{z}, \mathbf{y}), \mathbf{z}, \mathbf{y})-1)^{2}\\
\begin{aligned} L_{r e c} &=\|\mathbf{x}-G(E(\mathbf{x}, \mathbf{y}), \mathbf{y})\|^{2} \\ L_{K L} &=D_{K L}(p(\hat{\mathbf{z}} | \mathbf{x}, \mathbf{y}) \| q(\mathbf{z})) \end{aligned}
$$

The loss of encoder: reconstruction loss and Kullback-Leibler (KL) divergence loss



## Performance & Ablation Study
**Diversity**
![Content-aware Generative Modeling of Graphic Design Layouts - Zheng - SIGGRSPH 2019](https://i.imgur.com/xIu4TQJ.jpg)

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


**Constrained layout generation results**
In each case, the input con- tents and the input sketch that indicates the approximate positions and sizes of the desired elements in the output layouts (“T ”: Text element, “I ”: Image element, “H”: Headline element, “T\I”: Text-over-image element, “H \I ”: Headline-over-image element) are shown on the left. Results by the baseline (Baseline), our method (Ours), and the ground truth (Ground Truth) are shown on the right, where the Headline is filled with a sequence of A’s in bold. Note that, in each case, the text and image proportions used in both our method and the baseline are obtained from the ground truth layout. 
![Content-aware Generative Modeling of Graphic Design Layouts - Zheng - SIGGRSPH 2019](https://i.imgur.com/s3DtBJZ.jpg)

## Related
-  [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification - Salimans - ICLR 2017](https://arxivnote.ddlee.cn/PixelCNN++-Improving-the-PixelCNN-with-Discretized-Logistic-Mixture-Likelihood-and-Other-Modification.html)
- VQ-VAE-2: [ Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019](https://arxivnote.ddlee.cn/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2.html)