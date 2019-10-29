---
title: The Perception-Distortion Tradeoff - Blau - CVPR 2018 - Matlab
tag:
- Low-Level Vision
- Theoetical
redirect_from: /The-Perception-Distortion-Tradeoff-Blau-CVPR.html
---



## Info
- Title: **The Perception-Distortion Tradeoff**
- Task: **Low-level Vision**
- Author: Y. Blau and T. Michaeli
- Date:  Nov. 2017.
- Arxiv: [1711.06077](https://arxiv.org/abs/1711.06077)
- Published: CVPR 2018
- Affiliation: Technion-Israel Institute of Technology

## Highlights & Drawbacks
- Theoretical proof of Perception-Distortion tradeoff in low-level vision tasks like Super-Resolution.
- As mean distortion decreases, the optimal probability for correctly discriminating the restored image from original one must increase, indicating worse perceptual quality
- GANs provide a principled way to approach the perception-distortion bound


## Motivation & Design
![CleanShot 2019-08-17 at 16.01.35@2x](https://i.imgur.com/gFfPApr.jpg)

Problem setting: Given an original image $x ∼ p_X$ , a degraded image y is observed according to some conditional distribution $p_Y \|X $. Given the degraded image $y$, an estimate $x$ is constructed according to some conditional distribution $p_{\hat{X}} \|Y$ . Distortion is quantified by the mean of some distortion measure between Xˆ and X. The perceptual quality index corresponds to the deviation between $p_{\hat{X}}$  and $p_X​$.

The perception-distortion function of a signal restoration task is given by:
$$
P(D)=\min _{p_{X | Y}} d\left(p_{X}, p_{\hat{X}}\right) \quad \text { s.t. } \quad \mathbb{E}[\Delta(X, \hat{X})] \leq D
$$
where $∆(·, ·)$ is a distortion measure and $d(·, ·)$ is a divergence between distributions.

**The perception-distortion tradeoff)**. Assume the above problem setting, If $d(p,q)$ is convex in its second argument, then the perception-distortion function $P (D)$  is
1. monotonically non-increasing; 
2. convex.

![CleanShot 2019-08-17 at 16.00.56@2x](https://i.imgur.com/HnTCu7u.jpg)
Image restoration algorithms can be characterized by their average distortion and by the perceptual quality of the images they produce. We show that there exists a region in the perception-distortion plane which can- not be attained, regardless of the algorithmic scheme. When in proximity of this unattainable region, an algorithm can be potentially improved only in terms of its distortion or in terms of its perceptual quality, one at the expense of the other.



## Performance & Ablation Study
**Perception-distortion evaluation of SR algorithms** 
We plot 16 algorithms on the perception-distortion plane. Perception is measured by the recent NR metric by Ma et al. which is specifically designed for SR quality assessment. Distortion is measured by the common full-reference metrics RMSE, SSIM, MS-SSIM, IFC, VIF and VGG2,2. In all plots, the lower left corner is blank, revealing an unattainable region in the perception-distortion plane. In proximity of the unattainable region, an improvement in perceptual quality comes at the expense of higher distortion.

![CleanShot 2019-08-17 at 16.06.07@2x](https://i.imgur.com/dN9o5Ju.jpg)


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
- [Project Site](http://webee.technion.ac.il/people/tomermic/PerceptionDistortion/PD_tradeoff.htm)
- [Oral Presentation at CVPR 2018](https://youtu.be/_aXbGqdEkjk?t=39m43s)
- [Matlab](http://webee.technion.ac.il/people/tomermic/PerceptionDistortion/PD_tradeoff_code.zip)

## Related
- [Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff - Blau - ICML 2019](https://arxivnote.ddlee.cn/Rethinking-Lossy-Compression-The-Rate-Distortion-Perception-Tradeoff-Blau-ICML.html)