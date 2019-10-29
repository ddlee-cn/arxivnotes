---
title: "Deformable Kernels: Adapting Effective Receptive Fields for Object Deformation"
tag:
- CNN
- Theoretical

---

## Info

- Title: Deformable Kernels: Adapting Effective Receptive Fields for Object Deformation

- Author: Hang Gao, Xizhou Zhu , Steve Lin, Jifeng Dai3


- Date: Oct. 2019

- Arxiv: [1910.02940](https://arxiv.org/abs/1910.02940)

  

## Highlights

Resample the original kernel space towards recovering the deformation of objects. Directly adapting the ERF while leaving the receptive field untouched.



## Abstract

Convolutional networks are not aware of an object's geometric variations, which leads to inefficient utilization of model and data capacity. To overcome this issue, recent works on deformation modeling seek to spatially reconfigure the data towards a common arrangement such that semantic recognition suffers less from deformation. This is typically done by augmenting static operators with learned free-form sampling grids in the image space, dynamically tuned to the data and task for adapting the receptive field. Yet adapting the receptive field does not quite reach the actual goal -- what really matters to the network is the "effective" receptive field (ERF), which reflects how much each pixel contributes. It is thus natural to design other approaches to adapt the ERF directly during runtime. In this work, we instantiate one possible solution as Deformable Kernels (DKs), a family of novel and generic convolutional operators for handling object deformations by directly adapting the ERF while leaving the receptive field untouched. At the heart of our method is the ability to resample the original kernel space towards recovering the deformation of objects. This approach is justified with theoretical insights that the ERF is strictly determined by data sampling locations and kernel values. We implement DKs as generic drop-in replacements of rigid kernels and conduct a series of empirical studies whose results conform with our theories. Over several tasks and standard base models, our approach compares favorably against prior works that adapt during runtime. In addition, further experiments suggest a working mechanism orthogonal and complementary to previous works.


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



## Motivation & Design

We show how different 3 × 3 convolutions interact withdeformations of two images. Kernel spaces are visualized as flat 2D Gaussians. Each “+” indicates a computation between a pixel and a kernel value sampled from the data and kernel space. Their colors represent corresponding kernel values. 

- (a, b) Rigid kernels cannot adapt to specific deformations,
thus consuming large model and data capacity. 
- (c) Deformable Convolutions (Dai et al., 2017) reconfigure data towards common arrangement to counter the effects of geometric deformation. 
- (d)Our Deformable Kernels (DKs) instead resample kernels and, in effect, adapt kernel spaces while leaving the data untouched. 


Note that (b) and (c) share kernel values but sample different data locations, while (b) and (d) share data locations but sample different kernel values.



![Deformable Kernels](https://i.imgur.com/IZe2ZtE.png)



### 2D Conv

$$
\boldsymbol{O}_{j}=\sum_{\boldsymbol{k} \in \mathcal{K}} \boldsymbol{I}_{j+k} \boldsymbol{W}_{\boldsymbol{k}}
$$



### Deformable Convolution



$$
\boldsymbol{O}_{j}=\sum_{\boldsymbol{k} \in \mathcal{K}} \boldsymbol{I}_{j+\boldsymbol{k}+\Delta j} \boldsymbol{W}_{\boldsymbol{k}}
$$



the ERF of Deformable Convolution:
$$
\mathcal{R}_{\mathrm{DC}}^{(n)}(i ; j)=\sum_{k_{m} \in \mathcal{K}} \mathcal{R}^{(n)}\left(i ; j+k_{m}+\Delta j_{m}, k_{m}\right)
$$

### Deformable Kernel

$$
\boldsymbol{O}_{j}=\sum_{\boldsymbol{k} \in \mathcal{K}} \boldsymbol{I}_{j+k} \boldsymbol{W}_{\boldsymbol{k}+\Delta \boldsymbol{k}}
$$



the ERF of Deformable Kernel:
$$
\mathcal{R}_{\mathrm{DK}}^{(n)}(i ; j)=\sum_{k_{m} \in \mathcal{K}} \mathcal{R}^{(n)}\left(i ; j+k_{m}, k_{m}+\Delta k_{m}\right)
$$


### Global DK and Local DK


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



In practice, we implement Gglobal as a stack of one global average pooling layer, which reduces feature maps into a vector, and another fully-connected layer without non-linearities, which projects the reduced vector into an offset vector of 2K 2 dimensions. Then, we apply these offsets to all convolutions for the input image following Equation 7. For local DKs, we implement Glocal as an extra convolution that has the same configuration as the target kernel, except that it only has 2K 2 output channels. This produces kernel sampling offsets {∆k} that are additionally indexed by output locations j. It should be noted that similar designs were also discussed in Jia et al. (2016), in which filters are generated given either an image or individual patches from scratch rather than by resampling.



![Deformable Kernels](https://i.imgur.com/Q5fUZb3.png)



(a) The global DK learns one set of kernel sampling grid given an input image and apply it to all data positions. (b) The local DK adapts kernels for each input patches, and induces better locality for deformation modeling.






## Experiments & Ablation Study



### Image Classification

![Deformable Kernels](https://i.imgur.com/G2OUt7o.png)



### Object Detection



![Deformable Kernels](https://i.imgur.com/fZGXWl4.png)



## Related

- [Deformable Convolution in Object Detection: PyTorch Implementation(with CUDA)](https://cvnote.ddlee.cn/2019/09/19/Deformable-Conv-PyTorch.html)

- [CondConv: Conditionally Parameterized Convolutions for Efficient Inference](https://arxivnote.ddlee.cn/2019/10/15/CondConv-Conditionally-Parameterized-Convolutions-NIPS-2019.html)

- [Selective Kernel Networks](https://arxivnote.ddlee.cn/2019/10/14/Selective-Kernel-Networks-CVPR-2019.html)