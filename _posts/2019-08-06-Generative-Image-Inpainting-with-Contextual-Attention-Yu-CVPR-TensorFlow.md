---
title: Generative Image Inpainting with Contextual Attention - Yu - CVPR 2018 - TensorFlow
tag:
- GAN
- Attention
- Image Inpainting
redirect_from: /Generative-Image-Inpainting-with-Contextual-Attention-Yu-CVPR-TensorFlow.html
---



## Info
- Title: **Generative Image Inpainting with Contextual Attention**
- Task: **Image Inpainting**
- Author: J. Yu, Z. Lin, J. Yang, X. Shen, X. Lu, and T. S. Huang
- Date:  Jan. 2018
- Arxiv: [1801.07892](https://arxiv.org/abs/1801.07892)
- Published: CVPR 2018
- Affiliation: UIUC & Adobe

## Highlights & Drawbacks
- A novel contextual attention layer to explicitly attend on related feature patches at distant spatial locations.
- Introduce spatially discounted reconstruction loss to improve the training stability and speed based on the current the state-of-the-art generative image inpainting network


## Motivation & Design

Overview of our improved generative inpainting framework. The coarse network is trained with reconstruction loss explicitly, while the refinement network is trained with reconstruction loss, global and local WGAN-GP adversarial loss.

![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/Tw2ryjB.jpg)

Illustration of the contextual attention layer. Firstly we use convolution to compute matching score of foreground patches with background patches (as convolu- tional filters). Then we apply softmax to compare and get attention score for each pixel. Finally we reconstruct fore- ground patches with background patches by performing de- convolution on attention score. The contextual attention layer is differentiable and fully-convolutional.

![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/PvfMyPW.jpg)

Based on coarse result from the first encoder- decoder network, two parallel encoders are introduced and then merged to single decoder to get inpainting result. For visualization of attention map, color indicates relative loca- tion of the most interested background patch for each pixel in foreground. For examples, white (center of color coding map) means the pixel attends on itself, pink on bottom-left, green means on top-right.

![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/WOFEv0p.jpg)

**Training Procedure**
![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/8SAiQOs.jpg)

## Performance & Ablation Study

![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/gzKLIbb.jpg)

Based on coarse result from the first encoder- decoder network, two parallel encoders are introduced and then merged to single decoder to get inpainting result. For visualization of attention map, color indicates relative loca- tion of the most interested background patch for each pixel in foreground. For examples, white (center of color coding map) means the pixel attends on itself, pink on bottom-left, green means on top-right.

![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/8E4Qmg5.jpg)


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
- [Project Site](http://jiahuiyu.com/deepfill/)
- [TensorFlow(Official)](https://github.com/JiahuiYu/generative_inpainting)

## Related
- [Image Inpainting: From PatchMatch to Pluralistic](https://arxivnote.ddlee.cn/Imbalance-Problems-in-Object-Detection-A-Review-Oksuz-2019.html)
- [Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://arxivnote.ddlee.cn/Globally-and-locally-consistent-image-completion-SIGGRAPH.html)
- [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning - Nazeri - 2019 - PyTorch](https://arxivnote.ddlee.cn/EdgeConnect-Generative-Image-Inpainting-with-Adversarial-Edge-Learning-Nazeri.html)