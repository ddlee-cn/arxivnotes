---
title: "InstaGAN: Instance-aware Image-to-Image Translation - Sangwoo Mo - ICLR 2019"
tag:
- Image-to-Image Translation
- GAN
redirect_from: /InstaGAN-Instance-aware-Image-to-Image-Translation-Sangwoo-Mo-ICLR-2019.html
---



## Info
- Title: **InstaGAN: Instance-aware Image-to-Image Translation**
- Task: **Image-to-Image Translation**
- Author: Sangwoo Mo, Minsu Cho, Jinwoo Shin
- Date:  Dec. 2018
- Arxiv: [1812.10889](https://arxiv.org/abs/1812.10889)
- Published: ICLR 2019

## Highlights & Drawbacks
- Instance-level translation with semantic map
- Sequential mini-batch training strategy

## Abstract
Unsupervised image-to-image translation has gained considerable attention due to the recent impressive progress based on generative adversarial networks (GANs). However, previous methods often fail in challenging cases, in particular, when an image has multiple target instances and a translation task involves significant changes in shape, e.g., translating pants to skirts in fashion images. To tackle the issues, we propose a novel method, coined instance-aware GAN (InstaGAN), that incorporates the instance information (e.g., object segmentation masks) and improves multi-instance transfiguration. The proposed method translates both an image and the corresponding set of instance attributes while maintaining the permutation invariance property of the instances. To this end, we introduce a context preserving loss that encourages the network to learn the identity function outside of target instances. We also propose a sequential mini-batch inference/training technique that handles multiple instances with a limited GPU memory and enhances the network to generalize better for multiple instances. Our comparative evaluation demonstrates the effectiveness of the proposed method on different image datasets, in particular, in the aforementioned challenging cases.


## Motivation & Design
![InstaGAN: Instance-aware Image-to-Image Translation](https://i.imgur.com/LnP6Vx5.jpg)

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

(a) Overview of InstaGAN, where generators GXY, GYX and discriminator DX, DY follows the architectures in (b) and (c), respectively. Each network is designed to encode both an image and set of instance masks. G is permutation equivariant, and D is permutation invariant to the set order. To achieve properties, we sum features of all set elements for invariance, and then concatenate it with the identity mapping for equivariance.


![InstaGAN: Instance-aware Image-to-Image Translation](https://i.imgur.com/wPsmqRC.jpg)

Overview of the sequential mini-batch training with instance subsets (mini-batches) of size 1,2, and 1, as shown in the top right side. The content loss is applied to the intermediate samples of current mini-batch, and GAN loss is applied to the samples of aggregated mini-batches. We detach every iteration in training, in that the real line indicates the backpropagated paths and dashed lines indicates the detached paths. See text for details.

## Performance & Ablation Study

![InstaGAN: Instance-aware Image-to-Image Translation](https://i.imgur.com/8sdR7Fg.jpg)


![InstaGAN: Instance-aware Image-to-Image Translation](https://i.imgur.com/8oqUKQC.jpg)

Ablation study on the effect of each component of our method: the InstaGAN architecture, the context preserving loss, and the sequential mini-batch inference/training algorithm, which are denoted as InstaGAN, Lctx, and Sequential, respectively.

![InstaGAN: Instance-aware Image-to-Image Translation](https://i.imgur.com/jmmycKj.jpg)


Ablation study on the effects of the sequential mini-batch inference/training technique. The left and right side of title indicates which method used for training and inference, respectively, where “One” and “Seq” indicate the one-step and sequential schemes, respectively.


## Code
- [PyTorch](https://github.com/sangwoomo/instagan)

### Training Process

```python
def optimize_parameters(self):
    # init setting
    self.real_A_img_sng = self.real_A_img
    self.real_B_img_sng = self.real_B_img
    self.fake_A_seg_list = list()
    self.fake_B_seg_list = list()
    self.rec_A_seg_list = list()
    self.rec_B_seg_list = list()

    # sequential mini-batch translation
    for i in range(self.ins_iter):
        # forward
        self.forward(i)

        # G_A and G_B
        if self.forward_A or self.forward_B:
            self.set_requires_grad([self.netD_A, self.netD_B], False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        # D_A and D_B
        if self.forward_A or self.forward_B:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            if self.forward_A:
                self.backward_D_A()
            if self.forward_B:
                self.backward_D_B()
            self.optimizer_D.step()

        # update setting for next iteration
        self.real_A_img_sng = self.fake_B_img_sng.detach()
        self.real_B_img_sng = self.fake_A_img_sng.detach()
        self.fake_A_seg_list.append(self.fake_A_seg_sng.detach())
        self.fake_B_seg_list.append(self.fake_B_seg_sng.detach())
        self.rec_A_seg_list.append(self.rec_A_seg_sng.detach())
        self.rec_B_seg_list.append(self.rec_B_seg_sng.detach())
```



### Forward Pass

```python
def forward(self, idx=0):
    N = self.opt.ins_per
    self.real_A_seg_sng = self.real_A_segs[:, N*idx:N*(idx+1), :, :]  # ith mask
    self.real_B_seg_sng = self.real_B_segs[:, N*idx:N*(idx+1), :, :]  # ith mask
    empty = -torch.ones(self.real_A_seg_sng.size()).to(self.device)  # empty image

    self.forward_A = (self.real_A_seg_sng + 1).sum() > 0  # check if there are remaining instances
    self.forward_B = (self.real_B_seg_sng + 1).sum() > 0  # check if there are remaining instances

    # forward A
    if self.forward_A:
        self.real_A_sng = torch.cat([self.real_A_img_sng, self.real_A_seg_sng], dim=1)
        self.fake_B_sng = self.netG_A(self.real_A_sng)
        self.rec_A_sng = self.netG_B(self.fake_B_sng)

        self.fake_B_img_sng, self.fake_B_seg_sng = self.split(self.fake_B_sng)
        self.rec_A_img_sng, self.rec_A_seg_sng = self.split(self.rec_A_sng)
        fake_B_seg_list = self.fake_B_seg_list + [self.fake_B_seg_sng]  # not detach
        for i in range(self.ins_iter - idx - 1):
            fake_B_seg_list.append(empty)

        self.fake_B_seg_mul = torch.cat(fake_B_seg_list, dim=1)
        self.fake_B_mul = torch.cat([self.fake_B_img_sng, self.fake_B_seg_mul], dim=1)

    # forward B
    if self.forward_B:
        self.real_B_sng = torch.cat([self.real_B_img_sng, self.real_B_seg_sng], dim=1)
        self.fake_A_sng = self.netG_B(self.real_B_sng)
        self.rec_B_sng = self.netG_A(self.fake_A_sng)

        self.fake_A_img_sng, self.fake_A_seg_sng = self.split(self.fake_A_sng)
        self.rec_B_img_sng, self.rec_B_seg_sng = self.split(self.rec_B_sng)
        fake_A_seg_list = self.fake_A_seg_list + [self.fake_A_seg_sng]  # not detach
        for i in range(self.ins_iter - idx - 1):
            fake_A_seg_list.append(empty)

        self.fake_A_seg_mul = torch.cat(fake_A_seg_list, dim=1)
        self.fake_A_mul = torch.cat([self.fake_A_img_sng, self.fake_A_seg_mul], dim=1)
```



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


### Backward Pass for Generator

```python
def backward_G(self):
    lambda_A = self.opt.lambda_A
    lambda_B = self.opt.lambda_B
    lambda_idt = self.opt.lambda_idt
    lambda_ctx = self.opt.lambda_ctx

    # backward A
    if self.forward_A:
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B_mul), True)
        self.loss_cyc_A = self.criterionCyc(self.rec_A_sng, self.real_A_sng) * lambda_A
        self.loss_idt_B = self.criterionIdt(self.netG_B(self.real_A_sng), self.real_A_sng.detach()) * lambda_A * lambda_idt
        weight_A = self.get_weight_for_ctx(self.real_A_seg_sng, self.fake_B_seg_sng)
        self.loss_ctx_A = self.weighted_L1_loss(self.real_A_img_sng, self.fake_B_img_sng, weight=weight_A) * lambda_A * lambda_ctx
    else:
        self.loss_G_A = 0
        self.loss_cyc_A = 0
        self.loss_idt_B = 0
        self.loss_ctx_A = 0

    # backward B
    if self.forward_B:
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A_mul), True)
        self.loss_cyc_B = self.criterionCyc(self.rec_B_sng, self.real_B_sng) * lambda_B
        self.loss_idt_A = self.criterionIdt(self.netG_A(self.real_B_sng), self.real_B_sng.detach()) * lambda_B * lambda_idt
        weight_B = self.get_weight_for_ctx(self.real_B_seg_sng, self.fake_A_seg_sng)
        self.loss_ctx_B = self.weighted_L1_loss(self.real_B_img_sng, self.fake_A_img_sng, weight=weight_B) * lambda_B * lambda_ctx
    else:
        self.loss_G_B = 0
        self.loss_cyc_B = 0
        self.loss_idt_A = 0
        self.loss_ctx_B = 0

    # combined loss
    self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cyc_A + self.loss_cyc_B + self.loss_idt_A + self.loss_idt_B + self.loss_ctx_A + self.loss_ctx_B
    self.loss_G.backward()
```







### Backward Pass for Discriminator

```python
def backward_D_basic(self, netD, real, fake):
    # Real
    pred_real = netD(real)
    loss_D_real = self.criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = self.criterionGAN(pred_fake, False)
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    loss_D.backward()
    return loss_D

def backward_D_A(self):
    fake_B = self.fake_B_pool.query(self.fake_B_mul)
    self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

def backward_D_B(self):
    fake_A = self.fake_A_pool.query(self.fake_A_mul)
    self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
```







## Related

- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
