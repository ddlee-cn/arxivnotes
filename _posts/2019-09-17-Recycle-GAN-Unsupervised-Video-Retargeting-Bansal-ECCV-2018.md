---
title: "Recycle-GAN: Unsupervised Video Retargeting - Bansal - ECCV 2018"
tag:
- Video Translation
- GAN
---



## Info

- Title: Recycle-GAN: Unsupervised Video Retargeting
- Task: Video-to-Video Translation
- Author: Aayush Bansal, Shugao Ma, Deva Ramanan, and Yaser Sheikh
- Arxiv: [1808.05174](https://arxiv.org/abs/1808.05174)
- Published: ECCV 2018

## Highlights

Introduce a new approach that incorporates spatiotemporal cues with conditional generative adversarial networks for video retargeting. We demonstrate the advantages of spatiotemporal constraints over spatial constraints for image-to-labels, and labels-to-image in diverse environmental settings. We then present the proposed approach in learning better association between two domains, and its importance for self-supervised content alignment of the visual data. Inspired by the ever-existing nature of space-time, we qualitatively demonstrate the effectiveness of our approach for various natural processes

## Abstract

We introduce a data-driven approach for unsupervised video retargeting that translates content from one domain to another while preserving the style native to a domain, i.e., if contents of John Oliver's speech were to be transferred to Stephen Colbert, then the generated content/speech should be in Stephen Colbert's style. Our approach combines both spatial and temporal information along with adversarial losses for content translation and style preservation. In this work, we first study the advantages of using spatiotemporal constraints over spatial constraints for effective retargeting. We then demonstrate the proposed approach for the problems where information in both space and time matters such as face-to-face translation, flower-to-flower, wind and cloud synthesis, sunrise and sunset.

## Motivation & Design

### Main Idea

![Recycle-GAN: Unsupervised Video Retargeting - Bansal - ECCV 2018](https://i.imgur.com/VYdL5T5.jpg)

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


(a) Pix2Pix [23]: Paired data is available. A simple function (Eq. 1) can be
learnt via regression to map X → Y .

(b) Cycle-GAN [53]: The data is not paired
in this setting. Zhu et al. [53] proposed to use cycle-consistency loss (Eq. 3) to deal
with the problem of unpaired data. 

(c) Recycle-GAN: The approaches so far have
considered independent 2D images only. Suppose we have access to unpaired but or-
dered streams (x1, x2 , . . . , xt , . . .) and (y1 , y2 . . . , ys , . . .).

### Recurrent loss

We have so far considered the setting when static data is available. Instead, assume that we have access to unpaired but ordered streams (x1 , x2 , . . . , xt, . . .) and (y1 , y2 . . . , ys, . . .). Our motivating application is learning a mapping between two videos from different domains. One option is to ignore the stream indices, and treat the data as an unpaired and unordered collection of samples from X and Y (e.g., learn mappings between shuffled video frames). We demonstrate that much better mapping can be learnt by taking advantage of the temporal ordering. To describe our approach, we first introduce a recurrent temporal predictor PX that is trained to predict future samples in a stream given its past:

$$
L_{\tau}\left(P_{X}\right)=\sum_{t}\left\|x_{t+1}-P_{X}\left(x_{1 : t}\right)\right\|^{2}
$$

where we write x1:t = (x1 . . . xt ).

### Recycle loss

We use this temporal prediction model to define a new cycle loss
across domains and time (Fig. 3-c) which we refer as a recycle loss:

$$
L_{r}\left(G_{X}, G_{Y}, P_{Y}\right)=\sum_{t}\left\|x_{t+1}-G_{X}\left(P_{Y}\left(G_{Y}\left(x_{1 : t}\right)\right)\right)\right\|^{2}
$$

where GY (x1:t ) = (GY (x1 ), . . . , GY (xt)). Intuitively, the above loss requires sequences of frames to map back to themselves. We demonstrate that this is a much richer constraint when learning from unpaired data streams.

## Experiments & Ablation Study

![Recycle-GAN: Unsupervised Video Retargeting - Bansal - ECCV 2018](https://i.imgur.com/rCKo9Uz.jpg)


Face to Face: The top row shows multiple examples of face-to-face between John Oliver and Stephen Colbert using our approach. The bottom row shows example of translation from John Oliver to a cartoon character, Barack Obama to Donald Trump, and Martin Luther King Jr. (MLK) to Barack Obama. Without any input alignment or manual supervision, our approach could capture stylistic expressions for these public figures. As an example, John Oliver’s dimple while smiling, the shape of mouth characteristic of Donald Trump, and the facial mouth lines to and smile of Stephen
Colbert.

## Code

[Project Site]([http://www.cs.cmu.edu/~aayushb/Recycle-GAN/](http://www.cs.cmu.edu/~aayushb/Recycle-GAN/))

[PyTorch]([https://github.com/aayushbansal/Recycle-GAN](https://github.com/aayushbansal/Recycle-GAN))

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
