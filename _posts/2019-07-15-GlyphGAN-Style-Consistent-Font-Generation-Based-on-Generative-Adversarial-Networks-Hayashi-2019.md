---
title: "GlyphGAN: Style-Consistent Font Generation Based on Generative Adversarial Networks - Hayashi - 2019"
tag:
- GAN
- Application
redirect_from: /GlyphGAN-Style-Consistent-Font-Generation-Based-on-Generative-Adversarial-Networks-Hayashi-2019.html
---



## Info
- Title: **GlyphGAN: Style-Consistent Font Generation Based on Generative Adversarial Networks**
- Task: **Font Generation**
- Author: H. Hayashi, K. Abe, and S. Uchida
- Date: May 2019
- Arxiv: [1905.12502](https://arxiv.org/abs/1905.12502)

## Highlights & Drawbacks
- Two encode vectors for character and style, respectively

<!-- more -->

## Motivation & Design
The main frame work is a DCGAN, with Wasserstein distance as discriminator Loss. Notice that char-ID is encoded with a saperate one-hot vector as priors applied to random vector for generator.
![GlyphGAN: Style-Consistent Font Generation Based on Generative Adversarial Networks](https://i.imgur.com/deEmd13.png)

## Performance & Ablation Study

### legibility evaluation
![GlyphGAN: Style-Consistent Font Generation Based on Generative Adversarial Networks](https://i.imgur.com/6qnFXdy.png)

### Style Consistency
The metric of style consistency is defined as:
$$
C_{\mathrm{s}}=\frac{1}{N C} \sum_{n=1}^{N} \frac{1}{\overline{d}_{n}} \sum_{c=1}^{C}\left(d_{n, c}-\overline{d}_{n}\right)^{2},
$$
where $N$ is the number of generated images (we used $N = 10,000$ in this
experiment), $C$ is the number of character classes, i.e.,$ C = 26$, $d_{n, c}$ is the distance between the generated font and the nearest real font, and $\overline{d}$is the average of $d_{n, c}$ over $c$. The metric C is the averaged coefficient of variation of  $d_{n, c}$, and represents an intra-style variation of the generated font images. The lower $C_{\mathrm{s}}$ is, the higher style consistency is.

![GlyphGAN: Style-Consistent Font Generation Based on Generative Adversarial Networks](https://i.imgur.com/MUZ1Ccr.png)
