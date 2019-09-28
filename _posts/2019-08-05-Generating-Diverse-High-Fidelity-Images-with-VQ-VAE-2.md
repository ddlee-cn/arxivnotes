---
title: Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019
tag:
- VAE
- Image Generation
---



## Info
- Title: **Generating Diverse High-Fidelity Images with VQ-VAE-2**
- Task: **Image Generation**
- Author: A. Razavi, A. van den Oord, and O. Vinyals
- Date: Jun. 2019
- Arxiv: [1906.00446](https://arxiv.org/abs/1906.00446)
- Affiliation: Google DeepMind

## Highlights & Drawbacks
- Diverse generated results
- A multi-scale hierarchical organization of VQ-VAE
- Self-attention mechanism over autoregressive model


## Motivation & Design

![Generating Diverse High-Fidelity Images with VQ-VAE-2](https://i.imgur.com/kNEGBCj.png)

### Stage 1: Training hierarchical VQ-VAE
The design of hierarchical latent variables intends to separate local patterns (i.e., texture) from global information (i.e., object shapes). The training of the larger bottom level codebook is conditioned on the smaller top level code too, so that it does not have to learn everything from scratch.

![Generating Diverse High-Fidelity Images with VQ-VAE-2](https://i.imgur.com/HmBVGcm.png)


### Stage 2: Learning a prior over the latent discrete codebook 
The decoder can receive input vectors sampled from a similar distribution as the one in training. A powerful autoregressive model enhanced with multi-headed self-attention layers is used to capture the correlations in spatial locations that are far apart in the image with a larger receptive field. 

![Generating Diverse High-Fidelity Images with VQ-VAE-2](https://i.imgur.com/kbiYRcN.png)



## Performance & Ablation Study
Diverse Results, compared to BigGAN:
![Generating Diverse High-Fidelity Images with VQ-VAE-2](https://i.imgur.com/uJOzRRx.png)

Inception Score, FID, Precision-Recall Metric
![Generating Diverse High-Fidelity Images with VQ-VAE-2](https://i.imgur.com/qPTsco8.png)


## Related
- VQ-VAE: [Neural Discrete Representation Learning - van den Oord - NIPS 2017](https://arxivnote.ddlee.cn/Neural-Discrete-Representation-Learning.html)
- [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification - Salimans - ICLR 2017](https://arxivnote.ddlee.cn/PixelCNN++-Improving-the-PixelCNN-with-Discretized-Logistic-Mixture-Likelihood-and-Other-Modification.html)
- Gated PixelCNN: [Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016](https://arxivnote.ddlee.cn/Conditional-Image-Generation-with-PixelCNN-Decoders.html)
- PixelRNN & PixelCNN: [Pixel Recurrent Neural Networks - van den Oord - ICML 2016](https://arxivnote.ddlee.cn/Pixel-Recurrent-Neural-Networks.html)