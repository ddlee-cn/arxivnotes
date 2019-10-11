---
title: Seeing What a GAN Cannot Generate - Bau - ICCV 2019
tag:
- GAN
---

## Info
- Title: Seeing What a GAN Cannot Generate
- Author:
- Date: Sep. 2019
- [Paper](http://ganseeing.csail.mit.edu/papers/seeing.pdf)
- Published: ICCV 2019



## Abstract

Despite the success of Generative Adversarial Networks(GANs), mode collapse remains a serious issue during GAN training. To date, little work has focused on understanding and quantifying which modes have been dropped by a model. In this work, we visualize mode collapse at both the distribution level and the instance level. First, we deploy a semantic segmentation network to compare the distribution of segmented objects in the generated images with the target distribution in the training set. Differences in statistics reveal object classes that are omitted by a GAN. Second, given the identified omitted object classes, we visualize the GAN’s omissions directly. In particular, we compare specific differences between individual photos and their approximate inversions by a GAN. To this end, we relax the problem of inversion and solve the tractable problem of inverting a GAN layer instead of the entire generator. Finally, we use this framework to analyze several recent GANs trained on multiple datasets and identify their typical failure cases.



## Motivation & Design

![Seeing What a GAN Cannot Generate - Bau - ICCV 2019](https://i.imgur.com/jxfFCdU.png)

(a) We compare the distribution of object segmentations in the training set of LSUN churches to the distribution in the generated results: objects such as people, cars, and fences are dropped by the generator. 

(b) We compare pairs of a real image and its reconstruction in which individual instances of a person and a fence cannot be generated. In each block, we show a real photograph (top-left), a generated re-
construction (top-right), and segmentation maps for both (bottom).

### Generated Image Segmentation Statistics



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




The authors characterize omissions in the distribution as a whole, using Generated Image Segmentation Statistics:  segment both generated and ground truth images andcompare the distributions of segmented object classes. For example, the above figure shows that in a church GAN model,object classes such as people, cars, and fences appear on fewer pixels of the generated distribution as compared to the
training distribution.

#### Defining Fréchet Segmentation Distance (FSD)

It is an interpretable analog to the popular Fréchet Inception Distance
(FID) metric: 


$$
\mathrm{FSD} \equiv\left\|\mu_{g}-\mu_{t}\right\|^{2}+\operatorname{Tr}\left(\Sigma_{g}+\Sigma_{t}- 2(\Sigma_{g}\Sigma_{t})^{1/2}\right)
$$



In FSD formula, $\mu_{t}$ is the mean pixel count for each object class over a sample of training images,
and $\Sigma_{t}$ is the covariance of these pixel counts. Similarly, $\mu_{g}$ and $\Sigma_{g}$ reflect segmentation statistics for the generative model.



### Layer Inversion

Once omitted object classes are identified, the author want to visualize specific examples of failure cases. To do so, they must find image instances where the GAN should generate an object class but does not. We find such cases using a new reconstruction method called Layer Inversion which relaxes reconstruction to a tractable problem. Instead of inverting the entire GAN, their method inverts a layer of the generator.



![Seeing What a GAN Cannot Generate - Bau - ICCV 2019](https://i.imgur.com/KilUP0d.png)



First, train a network E to invert G; this is used to obtain an initial guess of the latent $z_0 = E(x)$ and its intermediate representation $r_0 = g_n (· · · (g_1 (z_0)))$. Then $r_0$ is used to initialize a search for $r^∗$ to obtain a reconstruction $x′ = G_f (r^∗)$ close to the target x.



## Experiments & Ablation Study

The paper examine the omissions of a GAN in two ways:

1. What does a GAN miss in its overall *distribution*?
2. What does a GAN miss in each *individual image*?

## Seeing Omissions in a GAN Distribution

To understand what the GAN's output distribution is missing, we gather segmentation statistics over the outputs, and compare the number of generated pixels in each output object class with the expected number in the training distribution.

A Progressive GAN trained to generate LSUN outdoor church images is analyzed below.


![Seeing What a GAN Cannot Generate - Bau - ICCV 2019](http://ganseeing.csail.mit.edu/img/progan-church-histogram.png)



This model does not generate enough pixels of people, cars, palm trees, or signboards compared to the training distribution.

Instead of drawing such complex objects, it draws too many pixels of simple things like earth and rivers and rock.


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



## Seeing Omissions in Individual GAN Images

Omissions in the distribution lead us to ask: how do these mistakes appear in individual images?

Seeing what a GAN *does not* generate requires us to compare the GAN's output with real photos. So instead of examining random images on their own, we use the GAN model to reconstruct real images from the training set. The differences reveal specific cases of what the GAN should ideally be able to draw, but cannot.



The GAN seems to avoid drawing people, synthesizing plausible scenes with the people removed.

GAN reconstruction              | Real photo 
:------------------------------:|:------------------------------:
![](http://ganseeing.csail.mit.edu/img/church_393_reconst.png) | ![](http://ganseeing.csail.mit.edu/img/church_393_target.png) 
![](http://ganseeing.csail.mit.edu/img/church_523_reconst.png) | ![](http://ganseeing.csail.mit.edu/img/church_523_target.png) 
![](http://ganseeing.csail.mit.edu/img/church_646_reconst.png) | ![](http://ganseeing.csail.mit.edu/img/church_646_target.png) 
![](http://ganseeing.csail.mit.edu/img/church_569_reconst.png) | ![](http://ganseeing.csail.mit.edu/img/church_569_target.png) 



A similar effect is seen for vehicles.



GAN reconstruction              | Real photo 
:------------------------------:|:------------------------------:
![](http://ganseeing.csail.mit.edu/img/church_54_reconst.png)  | ![](http://ganseeing.csail.mit.edu/img/church_54_target.png)
![](http://ganseeing.csail.mit.edu/img/church_666_reconst.png) | ![](http://ganseeing.csail.mit.edu/img/church_666_target.png)
![](http://ganseeing.csail.mit.edu/img/church_27_reconst.png)  | ![](http://ganseeing.csail.mit.edu/img/church_27_target.png)


## Code

[Project Site](http://ganseeing.csail.mit.edu/)

[PyTorch](https://github.com/davidbau/ganseeing)



## Related

