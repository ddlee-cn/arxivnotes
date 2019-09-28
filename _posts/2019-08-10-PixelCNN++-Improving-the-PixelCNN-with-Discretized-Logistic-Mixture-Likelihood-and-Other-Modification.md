---
title: "PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification - Salimans - ICLR 2017 - TensorFlow Code"
tag:
- Image Generation
- AutoRegressive
redirect_from: /PixelCNN++-Improving-the-PixelCNN-with-Discretized-Logistic-Mixture-Likelihood-and-Other-Modification.html
---



## Info
- Title: **PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications**
- Task: **Image Generation**
- Author: T. Salimans, A. Karpathy, X. Chen, and D. P. Kingma
- Date:  Jan. 2017
- Arxiv: [1701.05517](https://arxiv.org/abs/1701.05517)
- Published: ICLR 2017
- Affiliation: OpenAI

## Highlights & Drawbacks
- A discretized logistic mixture likelihood on the pixels, rather than a 256-way softmax, which speeds up training.
- Condition on whole pixels, rather than R/G/B sub-pixels, simplifying the model structure. 
- Downsampling to efficiently capture structure at multiple resolutions. 
- Additional shortcut connections to further speed up optimization.
- Regularize the model using dropout

## Motivation & Design
### Discretized logistic mixture likelihood
By choosing a simple continuous distribution for modeling $ν$ we obtain a smooth and memory efficient predictive distribution for $x$. Here, we take this continuous univariate distribution to be a mixture of logistic distributions which allows us to easily calculate the probability on the observed discretized value $x$ For all sub-pixel values $x$ excepting the edge cases 0 and 255 we have:
$$
\nu \sim \sum_{i=1}^{K} \pi_{i} \operatorname{logistic}\left(\mu_{i}, s_{i}\right)
$$

$$
P(x | \pi, \mu, s)=\sum_{i=1}^{K} \pi_{i}\left[\sigma\left(\left(x+0.5-\mu_{i}\right) / s_{i}\right)-\sigma\left(\left(x-0.5-\mu_{i}\right) / s_{i}\right)\right]
$$

The output of our network is thus of much lower dimension, yielding much denser gradients of the loss with respect to our parameters.


### More residual connections
![PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification](https://i.imgur.com/MN4a9m1.png)


## Performance & Ablation Study
![PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification](https://i.imgur.com/b715Tuc.png)

Training on a machine with 8 Maxwell TITAN X GPUs achieves 3.0 bits per dimension in about 10 hours and it takes approximately 5 days to converge to 2.92.

## Code
- [TensorFlow(Official)](https://github.com/openai/pixel-cnn)

## Related
- Gated PixelCNN: [Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016](https://arxivnote.ddlee.cn/Conditional-Image-Generation-with-PixelCNN-Decoders.html)
- PixelRNN & PixelCNN: [Pixel Recurrent Neural Networks - van den Oord - ICML 2016](https://arxivnote.ddlee.cn/Pixel-Recurrent-Neural-Networks.html)
- VQ-VAE: [Neural Discrete Representation Learning - van den Oord - NIPS 2017](https://arxivnote.ddlee.cn/Neural-Discrete-Representation-Learning.html)
- VQ-VAE-2: [ Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019](https://arxivnote.ddlee.cn/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2.html)