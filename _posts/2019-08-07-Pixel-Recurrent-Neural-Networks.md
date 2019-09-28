---
title: (PixelRNN & PixelCNN)Pixel Recurrent Neural Networks - van den Oord - ICML 2016
tag:
- Image Generation
- AutoRegressive
---



## Info
- Title: **Pixel Recurrent Neural Networks**
- Task: **Image Generation**
- Author: A. van den Oord, N. Kalchbrenner, and K. Kavukcuoglu
- Date:  Jan 2016
- Arxiv: [1601.06759](https://arxiv.org/abs/1601.06759)
- Published: ICML 2016(Best Paper Award)
- Affiliation: Google DeepMind

## Highlights & Drawbacks
- Fully tractable modeling of image distribution
- PixelRNN & PixelCNN


## Motivation & Design
To estimate the joint distribution $p(x)$ we write it as the product of the conditional distributions over the pixels:

$$
p(\mathbf{x})=\prod_{i=1}^{n^{2}} p\left(x_{i} | x_{1}, \ldots, x_{i-1}\right)
$$

Generating pixel-by-pixel with CNN, LSTM:
![Pixel Recurrent Neural Networks](https://i.imgur.com/Et9vL70.png)


## Performance & Ablation Study
Samples from ImageNet
![Pixel Recurrent Neural Networks](https://i.imgur.com/3OKYXbO.png)


## Related
- [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification - Salimans - ICLR 2017](https://arxivnote.ddlee.cn/PixelCNN++-Improving-the-PixelCNN-with-Discretized-Logistic-Mixture-Likelihood-and-Other-Modification.html)
- Gated PixelCNN: [Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016](https://arxivnote.ddlee.cn/Conditional-Image-Generation-with-PixelCNN-Decoders.html)
- VQ-VAE: [Neural Discrete Representation Learning - van den Oord - NIPS 2017](https://arxivnote.ddlee.cn/Neural-Discrete-Representation-Learning.html)
- VQ-VAE-2: [Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019](https://arxivnote.ddlee.cn/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2.html)