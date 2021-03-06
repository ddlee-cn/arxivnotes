---
title: Towards Instance-level Image-to-Image Translation - Shen - CVPR 2019
tag:
- Image-to-Image Translation
- GAN
redirect_from: /Towards-Instance-level-Image-to-Image-Translation-Shen-CVPR-2019.html
---



## Info
- Title: **Towards Instance-level Image-to-Image Translation**
- Task: **Image Translation**
- Author: Zhiqiang Shen, Mingyang Huang, Jianping Shi, Xiangyang Xue, Thomas Huang
- Date: May 2019
- Arxiv: [1905.01744](http://arxiv.org/abs/1905.01744)
- Published: CVPR 2019

## Highlights & Drawbacks
- The instance-level objective loss can help learn a more accurate reconstruction and incorporate diverse attributes of objects
- A more reasonable mapping: the styles used for target domain of local/global areas are from corresponding spatial regions in source domain.
- A large-scale, multimodal, highly varied Image-to-Image translation dataset, containing ∼155k streetscape images across four domains. 


<!-- more -->


## Motivation & Design
Disentangle background and object style in translation process:
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/AH9uHln.png)

The framework overview:
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/cMSETzP.png)

**Loss Design**
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/jps66rW.png)



![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/Ui9wDkn.png)


The instance-level translation dataset and comparisons with previous ones:
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/sgUndtd.png)


## Performance & Ablation Study
![Towards Instance-level Image-to-Image Translation](https://i.imgur.com/Rkibgwm.png)

The authors compared results with baselines like UNIT, CycleGAN, MUNIT and DRIT, using LPIPS distance to measure the diversity of generated images.

A visualization for generated style distribution is also provided.

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
[Project Site](http://zhiqiangshen.com/projects/INIT/index.html)



## Related

- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/2019/08/21/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE and INIT](https://arxivnote.ddlee.cn/2019/08/22/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)

- [TransGaGa: Geometry-Aware Unsupervised Image-to-Image Translation - Wayne Wu - CVPR 2019](https://arxivnote.ddlee.cn/2019/08/28/TransGaGa-Geometry-Aware-Unsupervised-Image-to-Image-Translation-Wayne-Wu-CVPR-2019.html)

- [InstaGAN: Instance-aware Image-to-Image Translation - Sangwoo Mo - ICLR 2019](https://arxivnote.ddlee.cn/2019/09/16/InstaGAN-Instance-aware-Image-to-Image-Translation-Sangwoo-Mo-ICLR-2019.html)