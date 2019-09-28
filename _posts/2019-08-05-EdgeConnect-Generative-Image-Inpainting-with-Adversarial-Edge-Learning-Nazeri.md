---
title: "EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning - Nazeri - 2019 - PyTorch"
tag:
- GAN
- Image Inpainting
---



## Info
- Title: **EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning**
- Task: **Image Inpainting**
- Author: K. Nazeri, E. Ng, T. Joseph, F. Qureshi, and M. Ebrahimi
- Date:  Jan. 2019
- Arxiv: [1901.00212](https://arxiv.org/abs/1901.00212)
- Published: ICCV 2019 Workshop

## Highlights & Drawbacks
- Interactive sketch editing for image completion
- A two-stage adversarial model that comprises of an edge generator followed by an image completion network.

## Motivation & Design

The spirit: “lines first, color next”, which is partly in- spired by our understanding of how artists work.

![EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://i.imgur.com/CHtT5JM.jpg)

### Edge Generator
Feature-Matching loss:
$$
\mathcal{L}_{F M}=\mathbb{E}\left[\sum_{i=1}^{L} \frac{1}{N_{i}}\left\|D_{1}^{(i)}\left(\mathbf{C}_{g t}\right)-D_{1}^{(i)}\left(\mathbf{C}_{p r e d}\right)\right\|_{1}\right]
$$

Adversarial loss:
$$
\begin{aligned} \mathcal{L}_{a d v, 1}=\mathbb{E}_{\left(\mathbf{C}_{g t}, \mathbf{I}_{g r a y}\right)}\left[\log D_{1}\left(\mathbf{C}_{g t}, \mathbf{I}_{g r a y}\right)\right] & ] \\ &+\mathbb{E}_{\mathbf{I}_{g r a y}} \log \left[1-D_{1}\left(\mathbf{C}_{p r e d}, \mathbf{I}_{g r a y}\right)\right] \end{aligned}
$$

### Completion Network
Adversarial loss:
$$
\begin{aligned} \mathcal{L}_{a d v, 2}=& \mathbb{E}_{\left(\mathbf{I}_{g t}, \mathbf{C}_{c o m p}\right)}\left[\log D_{2}\left(\mathbf{I}_{g t}, \mathbf{C}_{c o m p}\right)\right] \\ &+\mathbb{E}_{\mathbf{C}_{c o m p}} \log \left[1-D_{2}\left(\mathbf{I}_{p r e d}, \mathbf{C}_{c o m p}\right)\right] \end{aligned}
$$

Perceptual loss:
$$
\mathcal{L}_{p e r c}=\mathbb{E}\left[\sum_{i} \frac{1}{N_{i}}\left\|\phi_{i}\left(\mathbf{I}_{g t}\right)-\phi_{i}\left(\mathbf{I}_{p r e d}\right)\right\|_{1}\right]
$$

Style loss:
$$
\mathcal{L}_{s t y l e}=\mathbb{E}_{j}\left[\left\|G_{j}^{\phi}\left(\tilde{\mathbf{I}}_{p r e d}\right)-G_{j}^{\phi}\left(\tilde{\mathbf{I}}_{g t}\right)\right\|_{1}\right]
$$


## Performance & Ablation Study
### Quality Results
Left to Right: Original image, input image, generated edges, inpainted results without any post-processing.
![EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://i.imgur.com/gckghfQ.jpg)


### Quantitative results over Places2 with models
Left to right: Contextual Attention (CA), Globally and Locally Consistent Image Completion (GLCIC), Partial Convolu- tion (PConv) , G1 and G2 (Ours), G2 only with Canny edges (Canny). The best result of each row is boldfaced except for Canny. 
![EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://i.imgur.com/MXbesdU.jpg)

### Creative editing
![EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://i.imgur.com/P9JgUw8.jpg)

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
- [PyTorch](https://github.com/knazeri/edge-connect)

## Related
- [Image Inpainting: From PatchMatch to Pluralistic](https://arxivnote.ddlee.cn/Imbalance-Problems-in-Object-Detection-A-Review-Oksuz-2019.html)
- [Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://arxivnote.ddlee.cn/Globally-and-locally-consistent-image-completion-SIGGRAPH.html)
- [Generative Image Inpainting with Contextual Attention - Yu - CVPR 2018 - TensorFlow](https://arxivnote.ddlee.cn/Generative-Image-Inpainting-with-Contextual-Attention-Yu-CVPR-TensorFlow.html)