---
title: "Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019"
tag:
- Video Translation
- GAN
redirect_from: /Mocycle-GAN-Unpaired-Video-to-Video-Translation-Yang-Chen--ACMMM-2019.html
---



## Info
- Title: **Mocycle-GAN: Unpaired Video-to-Video Translation**
- Task: **Video-to-Video Translation**
- Author: Yang Chen, Yingwei Pan, Ting Yao, Xinmei Tian and Tao Mei
- Date:  Aug. 2019
- Arxiv: [1908.09514 ](https://arxiv.org/abs/1908.09514)
- Published: ACM MM 2019
- Affiliation: USTC!

## Abstract
Unsupervised image-to-image translation is the task of translating an image from one domain to another in the absence of any paired training examples and tends to be more applicable to practical applications. Nevertheless, the extension of such synthesis from image-to-image to video-to-video is not trivial especially when capturing spatio-temporal structures in videos. The difficulty originates from the aspect that not only the visual appearance in each frame but also motion between consecutive frames should be realistic and consistent across transformation. This motivates us to explore both appearance structure and temporal continuity in video synthesis. In this paper, we present a new Motion-guided Cycle GAN, dubbed as Mocycle-GAN, that novelly integrates motion estimation into unpaired video translator. Technically, Mocycle-GAN capitalizes on three types of constrains: adversarial constraint discriminating between synthetic and real frame, cycle consistency encouraging an inverse translation on both frame and motion, and motion translation validating the transfer of motion between consecutive frames. Extensive experiments are conducted on video-to-labels and labels-to-video translation, and superior results are reported when comparing to state-of-the-art methods. More remarkably, we qualitatively demonstrate our Mocycle-GAN for both flower-to-flower and ambient condition transfer.

## Motivation & Design

**Comparison between two unpaired translation approaches and Mocycle-GAN**

 ![Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019](https://i.imgur.com/wXxhqeR.jpg)


 (a) Cycle-GAN exploits cycle- consistency constraint to model appearance structure for unpaired image-to-image translation. 
(b) Recycle-GAN utilizes temporal predictor (PX and PY) to explore cycle consistency across both domains and time for unpaired video-to-video translation. 
(c) Mocycle-GAN explicitly models motion across frames with optical flow (fxt and fys ), and pursuits cycle consistency on motion that enforces the re- construction of motion. Motion translation is further exploited to transfer the motion across domains via motion translator(MX and MY ), strengthening the temporal continuity in video synthesis. Dot- ted line denotes consistency constraint between its two endpoints.

![Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019](https://i.imgur.com/RFqjulj.jpg)

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

The overview of Mocycle-GAN for unpaired video-to-video translation (X : source domain; Y : target domain). 
Note that here we only depict the forward cycle X → Y → X for simplicity. Mocycle-GAN consists of generators (GX and GY ) to synthesize frames across
domains, discriminators (DX and DY ) to distinguish real frames from synthetic ones, and motion translator (MX ) for motion translation
across domains. Given two real consecutive frames xt and xt+1, we firstly translate them into the synthetic frames x􏰂t and x􏰂t+1 via GX , which
are further transformed into the reconstructed frames x r e c and x r e c through the inverse mapping G . In addition, two optical flow fx and tt+1 Y t
fxrec are obtained by capitalizing on FlowNet to represent the motion before and after the forward cycle. 

During training, we leverage three kinds of spatial/temporal constrains to explore appearance structure and temporal continuity for video translation: 

1. Adversarial Constraint (LAdv ) ensures each synthetic frame realistic at appearance through adversarial learning;
2. Frame and Motion Cycle Consistency Constraint (LFC and LMC) encourage an inverse translation on both frames and motions;
3. Motion Translation Constraint(LMT) validates the transfer of motion across domains in video synthesis. Specifically, the motion translator MX converts the optical flow fxt in source to fxt in target, which will be utilized to further warp the synthetic frame x􏰂t to the subsequent frame W (fx , x􏰂t ). This constraint encourages the synthetic 􏰂subsequent frame x􏰂t+1 to be consistent with the warped version W(fx ,x􏰂t) in the traceable points, leading to pixel-wise temporal continuity.

**Motion Cycle Consistency Constraint**
![Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019](https://i.imgur.com/NHzxR16.jpg)

**Motion Translation Constraint**
![Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019](https://i.imgur.com/jTUWJfN.jpg)

**The Training Procedure**
![Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019](https://i.imgur.com/esqY8Pn.jpg)

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

## Performance & Ablation Study
![Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019](https://i.imgur.com/3LW5sQp.jpg)

Examples of (a) video-to-labels and (b) labels-to-video results in Viper dataset under various ambient conditions. The original inputs, the output results by different models, and the ground truth outputs are given.

Ablation study for each design (i.e., Motion Cycle Con- sistency (MC) and Motion Translation (MT)) in Mocycle-GAN for video-to-labels on Viper.

![Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019](https://i.imgur.com/SR37GCK.jpg)


Ablation study for each design (i.e., Motion Cycle Con- sistency (MC) and Motion Translation (MT)) in Mocycle-GAN for labels-to-video on Viper.

![Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen -  ACM MM 2019](https://i.imgur.com/jn2Fybe.jpg)



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
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)

