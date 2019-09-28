---
title: "Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff - Blau - ICML 2019"
tag:
- Low-Level Vision
- Theory
redirect_from: /Rethinking-Lossy-Compression-The-Rate-Distortion-Perception-Tradeoff-Blau-ICML.html
---



## Info
- Title: **Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff**
- Task: **Image Compression**
- Author: Y. Blau and T. Michaeli
- Date:  Jan. 2019.
- Arxiv: [1901.07821](https://arxiv.org/abs/1901.07821)
- Published: ICML 2019
- Affiliation: Technion-Israel Institute of Technology

## Highlights & Drawbacks
- A theoretical analysis of the Rate-Distortion-Perception function
- Restricting the perceptual quality to be high, generally leads to an elevation of the rate-distortion curve, thus necessitation a sacrifice in either rate or distortion.

## Motivation & Design
**Lossy Compression**
![Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff](https://i.imgur.com/LKNhgbH.jpg)
A source signal $X ∼ p_X$ is mapped into a coded sequence by an encoder and back into an estimated signal Xˆ by the decoder. Three desired properties are: (i) the coded sequence be compact (low bit rate); (ii) the reconstruction $\hat{X}$ be similar to the source $X$ on average (low distortion); (iii) the distribution $p_{\hat{X}}$ be similar to $p_X$, so that decoded signals are perceived as genuine source signals (good perceptual quality).

The (information) rate-distortion-perception function is defined as:
$$
\begin{aligned} R(D, P)=& \min _{p_{X | X}} I(X, \hat{X}) \\ & \quad \text { s.t. } \mathbb{E}[\Delta(X, \hat{X})] \leq D, d\left(p_{X}, p_{\hat{X}}\right) \leq P \end{aligned}
$$

![Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff](https://i.imgur.com/ed3I2ya.jpg)

Perception constrained rate-distortion curves for a Bernoulli source. Shannon’s rate-distortion function (dashed curve) characterizes the best achievable rate under any prescribed distortion level, yet does not ensure good perceptual quality. When constraining the perceptual quality index $d_{TV} (p_X , p_{\hat{X}})$ to be low (good quality), the rate-distortion function elevates (solid curves). This indicates that good perceptual quality must come at the cost of a higher rate and/or a higher distortion. Here $X ∼ Bern( \frac{1}{10} )$

![Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff](https://i.imgur.com/ht4VziF.jpg)

![Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff](https://i.imgur.com/dZGp8EE.jpg)

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

The rate-distortion-perception function of a Bernoulli source. (a) Equi-rate level sets depicted on the rate-distortion- perception function R(D, P ). At low bit-rates, the equi-rate lines curve substantially when approaching P = 0, displaying the increasing tradeoff between distortion and perceptual quality. (b) Cross sections of R(D, P ) along perception-distortion planes. Notice the tradeoff between perceptual quality and distortion, which becomes stronger at low bit-rates. (c) Cross sections of R(D, P ) along rate-perception planes. Note that at constant distortion, the perceptual quality can be improved by increasing the rate.

![Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff](https://i.imgur.com/4h7LVbW.jpg)

## Performance & Ablation Study

![Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff](https://i.imgur.com/dh8SrTO.jpg)

Perceptual lossy compression of MNIST digits. Left: Shannon’s rate-distortion curve (black) describes the lowest possible rate (bits per digit) as a function of distortion, but leads to low perceptual quality (high $d_W$ values), especially at low rates. When constraining the perceptual quality to be good (low $P$ values), the rate-distortion curve elevates, indicating that this comes at the cost of a higher rate and/or distortion. Right: Encoder-decoder outputs along Shannon’s rate-distortion curve and along two equi-perceptual-quality curves. As the rate decreases, the perceptual quality along Shannon’s curve degrades significantly. This is avoided when constraining the perceptual quality, which results in visually pleasing reconstructions, even at extremely low bit-rates. Notice that increased perceptually quality does not imply increased accuracy, as most reconstructions fail to preserve the digits’ identities at a 2-bit rate.

![Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff](https://i.imgur.com/swysTa7.jpg)

The rate-distortion-perception function of MNIST images. (a) Equi-rate lines plotted on $R(D, P )$ highlight the tradeoff between distortion and perceptual quality at any constant rate. (b) Cross sections of $R(D, P )$ along perception-distortion planes show that this tradeoff becomes stronger at low bit-rates. (c) Cross-sections of $R(D, P )$ along rate-perception planes highlght that at any constant distortion, the perceptual quality can be improved by increasing the rate.


## Code
- [Oral Presentation at ICML 2019](https://slideslive.com/38916897)

## Related
- [The Perception-Distortion Tradeoff - Blau - CVPR 2018](https://arxivnote.ddlee.cn/The-Perception-Distortion-Tradeoff-Blau-CVPR.html)