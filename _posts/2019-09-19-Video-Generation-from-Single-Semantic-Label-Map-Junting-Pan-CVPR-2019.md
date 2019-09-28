---
title: Video Generation from Single Semantic Label Map - Junting Pan - CVPR 2019
tag:
- Video Generation
- GAN
redirect_from: /Video-Generation-from-Single-Semantic-Label-Map-Junting-Pan-CVPR-2019.html
---



## Info
- Title: **Video Generation from Single Semantic Label Map**
- Task: **Video Generation**
- Author: Junting Pan, Chengyu Wang, Xu Jia,  Jing Shao, Lu Sheng, Junjie Yan, and Xiaogang Wang
- Date:  Mar. 2019
- Arxiv: [1903.04480](https://arxiv.org/abs/1903.04480)
- Published: CVPR 2019

## Abstract
This paper proposes the novel task of video generation conditioned on a SINGLE semantic label map, which provides a good balance between flexibility and quality in the generation process. Different from typical end-to-end approaches, which model both scene content and dynamics in a single step, we propose to decompose this difficult task into two sub-problems. As current image generation methods do better than video generation in terms of detail, we synthesize high quality content by only generating the first frame. Then we animate the scene based on its semantic meaning to obtain the temporally coherent video, giving us excellent results overall. We employ a cVAE for predicting optical flow as a beneficial intermediate step to generate a video sequence conditioned on the initial single frame. A semantic label map is integrated into the flow prediction module to achieve major improvements in the image-to-video generation process. Extensive experiments on the Cityscapes dataset show that our method outperforms all competing methods.

## Motivation & Design
![Video Generation from Single Semantic Label Map - Junting Pan - CVPR 2019](https://i.imgur.com/tqYgbH9.jpg)


Overall architecture of the proposed image-to-video generation network. It consists two components: a) Motion Encoder and b) Video Decoder. For any pair of bidirectional flow predictions, consistency check is computed only in non occluded areas.

![Video Generation from Single Semantic Label Map - Junting Pan - CVPR 2019](https://i.imgur.com/9ZAu7nP.jpg)

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

**cVAE**
During training, the encoder Q(z|V, I0) learns to match the standard normal distribution, N(0,I). When running inference, the cVAE will generate a video sequence from a given starting frame I0 and a latent variable z sampled from N(0,I) without the need of the motion encoder.

 We compute an occlusion mask to omit regions which are either occluded or missing in the generated frame so that the consistency check is only conducted on non-occluded regions. 

 With the predicted optical flow, we can directly produce future frames by warping the initial frame. However, the generated frames obtained solely by warping has inherent flaws, as some parts of the objects may not be visible in one frame but appears in another. To fill in the holes caused by either occlusion or objects entering or leaving the scene, we propose to add a post-processing network after frame warping. It takes a warped frame and its corresponding occlusion mask Ob as the input, and generates the refined frame.


To infer future motion of a object in a static frame, the model needs to understand the semantic category of that object and its interaction with other objects and background. 

Semantic sequence encoder. Each sequence en- coder only focuses on learning either foreground or back- ground motion.

![CleanShot 2019-09-22 at 13.46.14@2x](https://i.imgur.com/dhHjqwo.jpg)


## Performance & Ablation Study
 Comparisons with other competing baselines. Notice that vid2vid uses a sequence of semantic label maps while other methods only take one as input. 

 ![CleanShot 2019-09-22 at 13.46.50@2x](https://i.imgur.com/zna41cu.jpg)

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

Evaluated Datasets:

Cityscapes [6] consists of urban scene videos recorded from a car driving on the street. It con- tains 2,975 training, 500 validation and 1,525 test video se- quences, each containing 30 frames. The ground truth seman- tic segmentation mask is only available for the 20th frame of every video. We use DeepLabV3[5] to compute semantic segmentation maps for all frames, which are used for train- ing and testing. We train the model using all videos from the training set, and test it on the validation set. UCF101 [28] The dataset contains 13, 220 videos of 101 action classes. KTH Action dataset [17] consists of 600 videos of people performing one of the six actions(walking, jogging, running, boxing, handwaving, hand-clapping). KITTI [9] similar to Cityscpes was recorded from a car traversing streets.



## Code
- [TensorFlow](https://github.com/junting/seg2vid)



## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
