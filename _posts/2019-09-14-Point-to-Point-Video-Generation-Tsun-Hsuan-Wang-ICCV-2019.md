---
title: Point-to-Point Video Generation - Tsun-Hsuan Wang - ICCV 2019
tag:
- Video Generation
redirect_from: /Point-to-Point-Video-Generation-Tsun-Hsuan-Wang-ICCV-2019.html
---



## Info
- Title: **Point-to-Point Video Generation**
- Task: **Video Generation**
- Author: Tsun-Hsuan Wang, Yen-Chi Cheng, Chieh Hubert Lin, Hwann-Tzong Chen, Min Sun
- Date:  Apr. 2019
- Arxiv: [1904.02912](https://arxiv.org/abs/1904.02912)
- Published: ICCV 2019

## Abstract 
While image manipulation achieves tremendous breakthroughs (e.g., generating realistic faces) in recent years, video generation is much less explored and harder to control, which limits its applications in the real world. For instance, video editing requires temporal coherence across multiple clips and thus poses both start and end constraints within a video sequence. We introduce point-to-point video generation that controls the generation process with two control points: the targeted start- and end-frames. The task is challenging since the model not only generates a smooth transition of frames, but also plans ahead to ensure that the generated end-frame conforms to the targeted end-frame for videos of various length. We propose to maximize the modified variational lower bound of conditional data likelihood under a skip-frame training strategy. Our model can generate sequences such that their end-frame is consistent with the targeted end-frame without loss of quality and diversity. Extensive experiments are conducted on Stochastic Moving MNIST, Weizmann Human Action, and Human3.6M to evaluate the effectiveness of the proposed method. We demonstrate our method under a series of scenarios (e.g., dynamic length generation) and the qualitative results showcase the potential and merits of point-to-point generation. For project page, see https://zswang666.github.io/P2PVG-Project-Page/



## Motivation & Design
**An overview of the novel components of p2p generation**

![Point-to-Point Video Generation - Tsun-Hsuan Wang - ICCV 2019](https://i.imgur.com/PnkBiwn.jpg)

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


(a) Our model is a VAE consisting of posterior qφ, prior pψ , and generator pθ (all with an LSTM for temporal coherency). We use KL-divergence to encourage pψ to be similar to qφ. To control the generation, we encode the targeted end-frame xT into a global descriptor. Both qφ and pψ are computed by considering not only the input frame (xt or xt−1), but also the “global descriptor” and “time counter”. We further use the “alignment loss” to align the encoder and decoder latent space to reinforce the control point consistency. 

(b) Our skip-frame training has a probability to skip the input frame in each timestamp where the input will be ignored completely and the hidden state will not be propagated at all (see the dashed line). 

(c) The control point consistency is achieved by posing CPC loss on pψ without harming the reconstruction objective of qφ (highlighted in bold).

## Performance & Ablation Study
![Point-to-Point Video Generation - Tsun-Hsuan Wang - ICCV 2019](https://i.imgur.com/mz5fZod.jpg)

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

Given a pair of (orange) start- and (red) end-frames, we show various lengths generation on Weizmann and Hu- man3.6M (Number beneath each frame indicates the timestamp). Our model can achieve high-intermediate-diversity and targeted end-frame consistency while aware of various-length generation at the same time.


## Code
- [Project Site](https://zswang666.github.io/P2PVG-Project-Page)

## Related

- [Image Generation from Layout - Zhao - CVPR 2019](https://arxivnote.ddlee.cn/2019/08/24/Image-Generation-from-Layout-Zhao-CVPR-2019.html)
- [MoCoGAN: Decomposing Motion and Content for Video Generation - Tulyakov - CVPR 2018](https://arxivnote.ddlee.cn/2019/09/13/MoCoGAN-Decomposing-Motion-and-Content-for-Video-Generation-Tulyakov-CVPR-2018.html)
- [Recycle-GAN: Unsupervised Video Retargeting - Bansal - ECCV 2018](https://arxivnote.ddlee.cn/2019/09/17/Recycle-GAN-Unsupervised-Video-Retargeting-Bansal-ECCV-2018.html)
- [Video Generation from Single Semantic Label Map - Junting Pan - CVPR 2019](https://arxivnote.ddlee.cn/2019/09/19/Video-Generation-from-Single-Semantic-Label-Map-Junting-Pan-CVPR-2019.html)