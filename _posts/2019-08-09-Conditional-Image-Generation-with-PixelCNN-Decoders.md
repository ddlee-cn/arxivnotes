---
title: Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016 - TensorFlow & PyTorch Code
tag:
- Image Generation
- AutoRegressive
---



## Info
- Title: **Conditional Image Generation with PixelCNN Decoders**
- Task: **Image Generation**
- Author: A. van den Oord, N. Kalchbrenner, O. Vinyals, L. Espeholt, A. Graves, and K. Kavukcuoglu
- Date:  Jun. 2016
- Arxiv: [1606.05328](https://arxiv.org/abs/1606.05328)
- Published: NIPS 2016 
- Affiliation: Google DeepMind

## Highlights & Drawbacks
- Conditional with class labels or conv embeddings
- Can also serve as a powerful decoder


## Motivation & Design
Typically, to make sure the CNN can only use information about pixels above and to the left of the current pixel, the filters of the convolution in PixelCNN are masked. However, its computational cost rise rapidly when stacked.

The gated activation unit:
$$
\mathbf{y}=\tanh \left(W_{k, f} * \mathbf{x}\right) \odot \sigma\left(W_{k, g} * \mathbf{x}\right),
$$
where $σ$ is the sigmoid non-linearity, $k$ is the number of the layer, $⊙$ is the element-wise product and $∗$ is the convolution operator.

Add a high-level image description represented as a latent vector $h$:
$$
\mathbf{y}=\tanh \left(W_{k, f} * \mathbf{x}+V_{k, f}^{T} \mathbf{h}\right) \odot \sigma\left(W_{k, g} * \mathbf{x}+V_{k, g}^{T} \mathbf{h}\right)
$$

![Conditional Image Generation with PixelCNN Decoders](https://i.imgur.com/DTseuKt.png)

## Performance & Ablation Study

Class-conditioned
![Conditional Image Generation with PixelCNN Decoders](https://i.imgur.com/iHGfefW.png)

Latent Vector(Embedding learned by convolutional networks)
![Conditional Image Generation with PixelCNN Decoders](https://i.imgur.com/aLq0uMz.png)


## Code
- [TensorFlow](https://github.com/anantzoid/Conditional-PixelCNN-decoder)
- [PyTorch](https://github.com/j-min/PixelCNN)

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


## Related
- [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification - Salimans - ICLR 2017](https://arxivnote.ddlee.cn/PixelCNN++-Improving-the-PixelCNN-with-Discretized-Logistic-Mixture-Likelihood-and-Other-Modification.html)=
- PixelRNN & PixelCNN: [Pixel Recurrent Neural Networks - van den Oord - ICML 2016](https://arxivnote.ddlee.cn/Pixel-Recurrent-Neural-Networks.html)
- VQ-VAE: [Neural Discrete Representation Learning - van den Oord - NIPS 2017](https://arxivnote.ddlee.cn/Neural-Discrete-Representation-Learning.html)
- VQ-VAE-2: [ Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019](https://arxivnote.ddlee.cn/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2.html)