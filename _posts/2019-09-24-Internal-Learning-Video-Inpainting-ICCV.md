---
title: An Internal Learning Approach to Video Inpainting - Haotian Zhang - ICCV 2019
tag:
- Video Inpainting
- Low-Level Vision

---

## Info

- Title: An Internal Learning Approach to Video Inpainting
- Task:  Video Inpainting
- Author: Haotian Zhang, Long Mai, Ning Xu, Zhaowen Wang, JohnCollomosse,  Hailin Jin    
- Date: Sep. 2019
- Arxiv: [1909.07957](https://arxiv.org/abs/1909.07957)
- Published: ICCV 2019



## Highlights

- Trained on the input video(with holes) only
- Jointly synthesizing content in both appearance and motion domains

## Abstract

We propose a novel video inpainting algorithm that simultaneously hallucinates missing appearance and motion (optical flow) information, building upon the recent 'Deep Image Prior' (DIP) that exploits convolutional network architectures to enforce plausible texture in static images. In extending DIP to video we make two important contributions. First, we show that coherent video inpainting is possible without a priori training. We take a generative approach to inpainting based on internal (within-video) learning without reliance upon an external corpus of visual data to train a one-size-fits-all model for the large space of general videos. Second, we show that such a framework can jointly generate both appearance and flow, whilst exploiting these complementary modalities to ensure mutual consistency. We show that leveraging appearance statistics specific to each video achieves visually plausible results whilst handling the challenging problem of long-term consistency.



## Motivation & Design

![Internal Learning Video Inpainting](https://cs.stanford.edu/~haotianz/publications/video_inpainting/images/framework.png)

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


In this work, we approach video inpainting with an internal learning formulation. The general idea is to use the input video as the training data to learn a generative neural network ${G}\theta$ to generate each target frame Ii from a corresponding noise map Ii. The noise map Ii has one channel and shares the same spatial size with the input frame. We sample the input noise maps independently for each frame and fix them during training. The generative network $G{\theta}$ is trained to predict both frames Ii and optical flow maps F. The model is trained entirely on the input video (with holes) without any external data, optimizing the combination of the image generation loss Lr, perceptual loss Lp, flow generation loss L and consistency loss Lc.

**The Consistency Loss**

With the network jointly predicts images and flows, we define the image-flow consistency loss
to encourage the generated frames and the generated flows to constrain each other: the neighboring frames should be generated such that they are consistent with the predicted flow between them.
$$
L_{c}\left(\hat{I}_{j}, \hat{F}_{i, j}\right)=\left\|\left(1-M_{i, j}^{f}\right) \odot\left(\hat{I}_{j}\left(\hat{F}_{i, j}\right)-\hat{I}_{i}\right)\right\|_{2}^{2}
$$


${I}_{j}\left({F}_{i, j}\right)$ denotes where the warped version of the generated frame ${I}_{j}$ using the generated flow $F_{i,j} $ through backward warping. We constrain this loss only in the hole regions f using the inverse mask $1−M_{i,j}​$  to encourage the training to focus on propagating information inside the hole. We find this simple and intuitive loss term allows the network to learn the notion of flow and leverage it to propagate training signal across distant frames.





## Experiments & Ablation Study

![Internal Learning Video Inpainting](https://cs.stanford.edu/~haotianz/publications/video_inpainting/images/results.png)

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


In the below figure, the authors visualize the feature similarity between a reference patch and all the
other patches from neighboring locations and other frames. Specifically, we select a reference patch from the middle frame with patch size 32x32 matching the receptive field of the encoder, and calculate the cosine similarity between the features of the reference and other patches. The feature vector of a patch is given by the neuron responses at the corresponding 1x1 spatial location of the feature map. The patch similarity map is shown on the middle frame as well as 4 nearby frames, encoded in the alpha channel. A higher opacity patch indicates a higher feature similarity.  The learned feature is able to track the exact patch instead of only searching for visually similar patches.



![Internal Learning Video Inpainting](https://i.imgur.com/4hr6j3d.png)






## Code

[Project Site](https://cs.stanford.edu/~haotianz/publications/video_inpainting/)



## Related

- [Deep Flow-Guided Video Inpainting - CVPR 2019](https://arxivnote.ddlee.cn/2019/09/28/Flow-Guided-Video-Inpainting.html)
- [Image Inpainting: From PatchMatch to Pluralistic](https://arxivnote.ddlee.cn/2019/09/22/Image-Inpainting-PatchMatch-Edge-Connect-Partial-Conv.html)
- [Deep Image Prior - Ulyanov - CVPR 2018](https://arxivnote.ddlee.cn/2019/08/26/Deep-Image-Prior-Ulyanov-CVPR-2018.html)
- [Generative Image Inpainting with Contextual Attention - Yu - CVPR 2018 - TensorFlow](https://arxivnote.ddlee.cn/2019/08/06/Generative-Image-Inpainting-with-Contextual-Attention-Yu-CVPR-TensorFlow.html)
- [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning - Nazeri - 2019 - PyTorch](https://arxivnote.ddlee.cn/2019/08/05/EdgeConnect-Generative-Image-Inpainting-with-Adversarial-Edge-Learning-Nazeri.html)
- [Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://arxivnote.ddlee.cn/2019/08/04/Globally-and-locally-consistent-image-completion-SIGGRAPH.html)