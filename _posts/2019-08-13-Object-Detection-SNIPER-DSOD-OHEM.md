---
title: "Object Detection Must Reads(3): SNIP, SNIPER, OHEM, and DSOD"
tag:
- Object Detection
- Review
redirect_from: /object-detection-must-reads-snip-sniper-ohem-dsod.html
---



In [part 1](https://arxivnote.ddlee.cn/object-detectin-fast-rcnn-faster-rcnn-rfcn.html) and [part 2](https://arxivnote.ddlee.cn/object-detection-must-reads-yolo-retinanet.html) of object detection posts, we reviewed 1-stage and 2-stage object detectors. In this one, we introduce tricks aiming fast, accurate object detection works, including training strategy(SNIP & SNIPER), sampling strategy(OHEM) and scratch training(DSOD).









# An analysis of scale invariance in object detection - SNIP - Singh - CVPR 2018

### Info
- Title: **An analysis of scale invariance in object detection - SNIP**
- Task: **Object Detection**
- Author: B. Singh and L. S. Davis
- Date: Nov. 2017
- Arxiv: [1711.08189](https://arxiv.org/abs/1711.08189)
- Published: CVPR 2018

### Highlights & Drawbacks
- Training strategy optimization, ready to integrate with other tricks
- Informing experiments for multi-scale training trick

<!-- more -->

### Design

The process of SNIP:
1. Select 3 image resolutions: (480, 800) to train [120, ∞) proposals, (800, 1200) to train [40, 160] proposals, (1400, 2000) to train [0, 80] for proposals

2. For each resolution image, BP only returns the gradient of the proposal within the corresponding scale.

3. This ensures that only one network is used, but the size of each training object is the same, and the size of the object of ImageNet is consistent to solve the problem of domain shift, and it is consistent with the experience of the backbone, and the training and test dimensions are consistent, satisfying " ImageNet pre-trained size, an object size, a network, a receptive field, these four match each other, and the train and test dimensions are the same.

4. A network, but using all the object training, compared to the scale specific detector, SNIP is fully conducive to the data

5. During the test, the same detector is measured once on each of the three resolution images, and only the detected boxes of the corresponding scale are retained for each resolution image, and then merged to execute SoftNMS.

## Performance & Ablation Study

The authors conducted experiments for RFCN and Faster-RCNN and SNIP improves performance for both meta architectures.

![An analysis of scale invariance in object detection](https://i.imgur.com/6gKN6kw.png)

Check full introduction at [An analysis of scale invariance in object detection - SNIP - Singh - CVPR 2018](https://arxivnote.ddlee.cn/An-analysis-of-scale-invariance-in-object-detection-SNIP-Singh-CVPR-2018.html).

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

## SNIPER: efficient multi-scale training - Singh - NIPS 2018 - MXNet Code

### Info
- Title: **SNIPER: efficient multi-scale training**
- Task: **Object Detection**
- Author: B. Singh, M. Najibi, and L. S. Davis
- Date: May 2018
- Arxiv: [1805.09300](https://arxiv.org/abs/1805.09300)
- Published: NIPS 2018

### Highlights & Drawbacks
- Efficient version of [SNIP](https://arxivnote.ddlee.cn/An-analysis-of-scale-invariance-in-object-detection-SNIP-Singh-CVPR-2018.html) training strategy for object detection
- Select ROIs with proper size only inside a batch


### Design
![SNIPER: efficient multi-scale training](https://i.imgur.com/fcqlVp8.gif)

Following [SNIP](https://ddleenote.blogspot.com/2019/05/an-analysis-of-scale-invariance-in.html), the authors put crops of an image which contain objects to be detected(called chips) into training instead of the entire image. This design also makes large-batch training possible, which accelerates the training process. This training method utilizes the context of the object, which can save unnecessary calculations for simple background(such as the sky) so that the utilization rate of training data is improved. 

![SNIPER: efficient multi-scale training](https://i.imgur.com/WUTZeG1.png)

The core design of SNIPER is the selection strategy for ROIs from a chip(a crop of entire image). The authors use several hyper-params to filter boxes with proper size in a batch, hopping that the detector network only learns features beyond object size.

Due to its memory efficient design, SNIPER can benefit from Batch Normalization during training and it makes larger batch-sizes possible for instance-level recognition tasks on a single GPU. Hence, there is no need to synchronize batch-normalization statistics across GPUs.

### Performance & Ablation Study
An improvement of the accuracy of small-size objects was reported according to the author’s experiments. 

![SNIPER: efficient multi-scale training](https://i.imgur.com/vQ3Qan1.png)


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
[MXNet](https://github.com/mahyarnajibi/SNIPER)



## (OHEM)Training Region-based Object Detectors with Online Hard Example Mining - Shrivastava et al. - CVPR 2016

### Info
- Title: **Training Region-based Object Detectors with Online Hard Example Mining**
- Task: **Object Detection**
- Author: A. Shrivastava, A. Gupta, and R. Girshick
- Date: Apr. 2016
- Arxiv: [1604.03540](https://arxiv.org/abs/1604.03540)
- Published: CVPR 2016

### Highlights & Drawbacks
- Learning-based design for balancing examples for ROI in 2-stage detection network
- Plug-in ready trick, easy to be integrated
- Additional Parameters for Training

### Motivation & Design

There is a 1:3 strategy in Faster-RCNN network, which samples negative ROIs(backgrounds) to balance the ratio for positive and negative data in a batch. It's empirical and hand-designed(need additional effort when setting hyper-params).

![ (OHEM)Training Region-based Object Detectors with Online Hard Example Mining](https://i.imgur.com/6aFz3zx.png)

The authors designed an additional sub-network to "learn" the sampling process for negative ROIs, forcing the network focus on ones which are similar to objects(the hard ones), such as backgrounds contain part of objects.

The 'hard' examples are defined using probability from detection head, which means that the sample network is exactly the classification network. In practice, the selecting range is set to [0.1, 0.5].

## Performance & Ablation Study

![ (OHEM)Training Region-based Object Detectors with Online Hard Example Mining](https://i.imgur.com/UyenHVl.png)


![ (OHEM)Training Region-based Object Detectors with Online Hard Example Mining](https://i.imgur.com/ETa4rjl.png)

OHEM can improve performance even after adding bells and whistles like Multi-scale training and Iterative bbox regression.

## Code
[caffe](https://github.com/abhi2610/ohem)


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

# DSOD: learning deeply supervised object detectors from scratch - Shen - ICCV 2017 - Caffe Code

## Info
- Title: **DSOD: learning deeply supervised object detectors from scratch**
- Task: **Object Detection**
- Author: Z. Shen, Z. Liu, J. Li, Y. Jiang, Y. Chen, and X. Xue
- Date: Aug. 2017
- Arxiv: [1708.01241](https://arxiv.org/abs/1708.01241)
- Published: ICCV 2017

## Highlights & Drawbacks
- Object Detection without pre-training
- DenseNet-like network

## Design

A common practice that used in earlier works such as R-CNN is to pre-train a backbone network on a categorical dataset like ImageNet, and then use these pre-trained weights as initialization of detection model. Although I have once successfully trained a small detection network from random initialization on a large dataset, there are few models trained from scratch when the number of instances in a dataset is limited like Pascal VOC and COCO. Actually, using better pre-trained weights is one of the tricks in detection challenges. DSOD attempts to train the detection network from scratch with the help of "Deep Supervision" from DenseNet.

The 4 principles authors argued for object detection networks:

    1. Proposal-free
    2. Deep supervision
    3. Stem Block
    4. Dense Prediction Structure

![DSOD: learning deeply supervised object detectors from scratch](https://i.imgur.com/amvcbcK.png)


## Performance & Ablation Study

 DSOD outperforms detectors with pre-trained weights.
![DSOD: learning deeply supervised object detectors from scratch](https://i.imgur.com/1dt4lad.png)

Ablation Study on parts:
![DSOD: learning deeply supervised object detectors from scratch](https://i.imgur.com/vKRUrAf.png)

## Code

[Caffe](https://github.com/szq0214/DSOD)


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
- [Object Detection Must Reads(1): Fast RCNN, Faster RCNN, R-FCN and FPN](https://arxivnote.ddlee.cn/object-detectin-fast-rcnn-faster-rcnn-rfcn.html)
- [Object Detection Must Reads(2): YOLO, YOLO9000, and RetinaNet](https://arxivnote.ddlee.cn/object-detection-must-reads-yolo-retinanet.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)
- [Convolutional Neural Network Must Reads: Xception, ShuffleNet, ResNeXt and DenseNet](https://arxivnote.ddlee.cn/convolutional-neural-network-xception-shufflenet-resnext-densenet.html)
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)