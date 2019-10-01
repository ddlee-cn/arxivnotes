---
title: "Recycle-GAN: Unsupervised Video Retargeting - Bansal - ECCV 2018"
tag:
- Video Translation
- GAN
redirect_from: /Recycle-GAN-Unsupervised-Video-Retargeting-Bansal-ECCV-2018.html
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

### Training Process

```python
def optimize_parameters(self):
    # forward
    self.forward()
    # G_A and G_B
    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()
    # D_A
    self.optimizer_D_A.zero_grad()
    self.backward_D_A()
    self.optimizer_D_A.step()
    # D_B
    self.optimizer_D_B.zero_grad()
    self.backward_D_B()
    self.optimizer_D_B.step()
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



#### Backward Pass for Discriminator

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
    fake_B0 = self.fake_B_pool.query(self.fake_B0)
    loss_D_A0 = self.backward_D_basic(self.netD_A, self.real_B0, fake_B0)

    fake_B1 = self.fake_B_pool.query(self.fake_B1)
    loss_D_A1 = self.backward_D_basic(self.netD_A, self.real_B1, fake_B1)

    fake_B2 = self.fake_B_pool.query(self.fake_B2)
    loss_D_A2 = self.backward_D_basic(self.netD_A, self.real_B2, fake_B2)

pred_B = self.fake_B_pool.query(self.pred_B2)
loss_D_A3 = self.backward_D_basic(self.netD_A, self.real_B2, pred_B)

    self.loss_D_A = loss_D_A0.data[0] + loss_D_A1.data[0] + loss_D_A2.data[0] + loss_D_A3.data[0]

def backward_D_B(self):
    fake_A0 = self.fake_A_pool.query(self.fake_A0)
    loss_D_B0 = self.backward_D_basic(self.netD_B, self.real_A0, fake_A0)

    fake_A1 = self.fake_A_pool.query(self.fake_A1)
    loss_D_B1 = self.backward_D_basic(self.netD_B, self.real_A1, fake_A1)

    fake_A2 = self.fake_A_pool.query(self.fake_A2)
    loss_D_B2 = self.backward_D_basic(self.netD_B, self.real_A2, fake_A2)

    pred_A = self.fake_A_pool.query(self.pred_A2)
    loss_D_B3 = self.backward_D_basic(self.netD_B, self.real_A2, pred_A)

    self.loss_D_B = loss_D_B0.data[0] + loss_D_B1.data[0] + loss_D_B2.data[0] + loss_D_B3.data[0]

```



#### Backward Pass for Generator

```python
def backward_G(self):
    lambda_idt = self.opt.identity
    lambda_A = self.opt.lambda_A
    lambda_B = self.opt.lambda_B
    # Identity loss
    if lambda_idt > 0:
        # G_A should be identity if real_B is fed.
        idt_A0 = self.netG_A(self.real_B0)
        idt_A1 = self.netG_A(self.real_B1)
        loss_idt_A = (self.criterionIdt(idt_A0, self.real_B0) + self.criterionIdt(idt_A1, self.real_B1) )* lambda_B * lambda_idt
        # G_B should be identity if real_A is fed.
        idt_B0 = self.netG_B(self.real_A0)
        idt_B1 = self.netG_B(self.real_A1)
        loss_idt_B = (self.criterionIdt(idt_B0, self.real_A0) + self.criterionIdt(idt_B1, self.real_A1)) * lambda_A * lambda_idt

        self.idt_A = idt_A0.data
        self.idt_B = idt_B0.data
        self.loss_idt_A = loss_idt_A.data[0]
        self.loss_idt_B = loss_idt_B.data[0]

    else:
        loss_idt_A = 0
        loss_idt_B = 0
        self.loss_idt_A = 0
        self.loss_idt_B = 0

    # GAN loss D_A(G_A(A))
    fake_B0 = self.netG_A(self.real_A0)
    pred_fake = self.netD_A(fake_B0)
    loss_G_A0 = self.criterionGAN(pred_fake, True)

    fake_B1 = self.netG_A(self.real_A1)
    pred_fake = self.netD_A(fake_B1)
    loss_G_A1 = self.criterionGAN(pred_fake, True)

#fake_B2 = self.netP_B(torch.cat((fake_B0,fake_B1),1))
if self.which_model_netP == 'prediction':
    fake_B2 = self.netP_B(fake_B0,fake_B1)
else:
    fake_B2 = self.netP_B(torch.cat((fake_B0,fake_B1),1))

pred_fake = self.netD_A(fake_B2)
loss_G_A2 = self.criterionGAN(pred_fake, True)

    # GAN loss D_B(G_B(B))
    fake_A0 = self.netG_B(self.real_B0)
    pred_fake = self.netD_B(fake_A0)
    loss_G_B0 = self.criterionGAN(pred_fake, True)

    fake_A1 = self.netG_B(self.real_B1)
    pred_fake = self.netD_B(fake_A1)
    loss_G_B1 = self.criterionGAN(pred_fake, True)

    #fake_A2 = self.netP_A(torch.cat((fake_A0,fake_A1),1))
if self.which_model_netP == 'prediction':
    fake_A2 = self.netP_A(fake_A0,fake_A1)
else:
    fake_A2 = self.netP_A(torch.cat((fake_A0,fake_A1),1))

    pred_fake = self.netD_B(fake_A2)
    loss_G_B2 = self.criterionGAN(pred_fake, True)

# prediction loss -- 
#pred_A2 = self.netP_A(torch.cat((self.real_A0, self.real_A1),1))
if self.which_model_netP == 'prediction':
    pred_A2 = self.netP_A(self.real_A0, self.real_A1)
else:
    pred_A2 = self.netP_A(torch.cat((self.real_A0, self.real_A1),1))

loss_pred_A = self.criterionCycle(pred_A2, self.real_A2) * lambda_A

#pred_B2 = self.netP_B(torch.cat((self.real_B0, self.real_B1),1))
if self.which_model_netP == 'prediction':
    pred_B2 = self.netP_B(self.real_B0, self.real_B1)
else:
    pred_B2 = self.netP_B(torch.cat((self.real_B0, self.real_B1),1))

loss_pred_B = self.criterionCycle(pred_B2, self.real_B2) * lambda_B

    # Forward recycle loss
    rec_A = self.netG_B(fake_B2)
    loss_recycle_A = self.criterionCycle(rec_A, self.real_A2) * lambda_A

    # Backward recycle loss
    rec_B = self.netG_A(fake_A2)
    loss_recycle_B = self.criterionCycle(rec_B, self.real_B2) * lambda_B

    # Fwd cycle loss 
    rec_A0 = self.netG_B(fake_B0)
    loss_cycle_A0 = self.criterionCycle(rec_A0, self.real_A0) * lambda_A

    rec_A1 = self.netG_B(fake_B1)
    loss_cycle_A1 = self.criterionCycle(rec_A1, self.real_A1) * lambda_A

    rec_B0 = self.netG_A(fake_A0)
    loss_cycle_B0 = self.criterionCycle(rec_B0, self.real_B0) * lambda_B

    rec_B1 = self.netG_A(fake_A1)
    loss_cycle_B1 = self.criterionCycle(rec_B1, self.real_B1) * lambda_B

    # combined loss
    loss_G = loss_G_A0 + loss_G_A1 + loss_G_A2 + loss_G_B0 + loss_G_B1 + loss_G_B2 + loss_recycle_A + loss_recycle_B + loss_pred_A + loss_pred_B + loss_idt_A + loss_idt_B + loss_cycle_A0 + loss_cycle_A1 + loss_cycle_B0 + loss_cycle_B1
    loss_G.backward()

```



## Related

- [Mocycle-GAN: Unpaired Video-to-Video Translation - Yang Chen - ACM MM 2019](https://arxivnote.ddlee.cn/2019/09/20/Mocycle-GAN-Unpaired-Video-to-Video-Translation-Yang-Chen-ACMMM-2019.html)

- [Video Generation from Single Semantic Label Map - Junting Pan - CVPR 2019](https://arxivnote.ddlee.cn/2019/09/19/Video-Generation-from-Single-Semantic-Label-Map-Junting-Pan-CVPR-2019.html)

- [MoCoGAN: Decomposing Motion and Content for Video Generation - Tulyakov - CVPR 2018](https://arxivnote.ddlee.cn/2019/09/13/MoCoGAN-Decomposing-Motion-and-Content-for-Video-Generation-Tulyakov-CVPR-2018.html)

- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/2019/08/21/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE and INIT](https://arxivnote.ddlee.cn/2019/08/22/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [PyTorch Code for vid2vid](https://cvnote.ddlee.cn/2019/09/06/vid2vid-PyTorch-GitHub.html)
- [PyTorch Code for CycleGAN](https://cvnote.ddlee.cn/2019/09/02/CycleGAN-PyTorch-GitHub.html)