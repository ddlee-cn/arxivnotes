---
title: Unsupervised Person Image Generation with Semantic Parsing Transformation - Sijie Song - CVPR 2019
tag:
- Image Generation
- GAN
redirect_from: /Unsupervised-Person-Image-Generation-with-Semantic-Parsing-Transformation-Sijie-Song-CVPR-2019.html
---



## Info

- Title: Unsupervised Person Image Generation with Semantic Parsing Transformation
- Task: Image Generation
- Author: Sijie Song, Wei Zhang, Jiaying Liu , Tao Mei
- Arxiv: [1829](https://arxiv.org/abs/)
- Published: CVPR 2019

## Highlights

We propose to address the unsupervised person image generation problem. Consequently, the problem is decomposed into semantic parsing transformation (HS ) and appearance generation (HA ).

We design a delicate training schema to carefully optimize HS and HA in an end-to-end manner, which generates better semantic maps and further improves the pose-guided image generation results.

## Abstract

In this paper, we address unsupervised pose-guided person image generation, which is known challenging due to non-rigid deformation. Unlike previous methods learning a rock-hard direct mapping between human bodies, we propose a new pathway to decompose the hard mapping into two more accessible subtasks, namely, semantic parsing transformation and appearance generation. Firstly, a semantic generative network is proposed to transform between semantic parsing maps, in order to simplify the non-rigid deformation learning. Secondly, an appearance generative network learns to synthesize semantic-aware textures. Thirdly, we demonstrate that training our framework in an end-to-end manner further refines the semantic maps and final results accordingly. Our method is generalizable to other semantic-aware person image generation tasks, eg, clothing texture transfer and controlled image manipulation. Experimental results demonstrate the superiority of our method on DeepFashion and Market-1501 datasets, especially in keeping the clothing attributes and better body shapes.

## Motivation & Design
![](https://i.imgur.com/6hxwcmj.png)

Semantic parsing transformation module aims to first generate a semantic map under the target pose, which provides crucial prior for the human body shape and clothing attributes. Guided by the predicted semantic map and the reference image, appearance generation module then synthesizes textures for the final output image.

![](https://i.imgur.com/DpZlzgl.png)

![](https://i.imgur.com/9xupXQH.png)


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

## Experiments & Ablation Study
![](https://i.imgur.com/NNH4yVN.png)


![](https://i.imgur.com/tgJyYSp.png)

## Code

[PyTorch]([https://github.com/SijieSong/person_generation_spt](https://github.com/SijieSong/person_generation_spt))

![](https://i.imgur.com/xGhPwvD.png)


```
    class Global_Generator(nn.Module):
        def __init__(self, input_nc, pose_dim, image_size, nfilters_enc, nfilters_dec, warp_skip, use_input_pose=True):
            super(Global_Generator, self).__init__()
            self.input_nc = input_nc
            # number of skip connections
            self.num_skips = 1 if warp_skip=='None' else 2
            self.warp_skip = warp_skip
            self.pose_dim = pose_dim
            self.nfilters_dec = nfilters_dec
            self.nfilters_enc = nfilters_enc
            self.image_size = image_size
            self.use_input_pose = use_input_pose
            # input parsing result to encoder_pose
            self.encoder_app = encoder(input_nc-self.pose_dim - 9, nfilters_enc)
            self.encoder_pose = encoder(self.pose_dim + 9, nfilters_enc)
    
            self.decoder = decoder(nfilters_dec, nfilters_enc, self.num_skips)
            self.pose_dim = 18
    
    
        def get_imgpose(self, input, use_input_pose, pose_dim):
            inp_img = input[:, :12] # include pose and parsing
            inp_pose = input[:, 12:12 + pose_dim] if use_input_pose else None
    
            tg_parsing = input[:, 12+pose_dim: 21+pose_dim] # target parsing
            tg_pose_index = 21 + pose_dim if use_input_pose else 6
            tg_pose = input[:, tg_pose_index:]
            
            return inp_img, inp_pose, tg_parsing, tg_pose
    
        def forward(self, input, warps, masks, fade_in_app, fade_in_pose, fade_in_alpha):
            
            inp_app, inp_pose, tg_parsing, tg_pose = self.get_imgpose(input, self.use_input_pose, self.pose_dim)
            inp_app = torch.cat([inp_app, inp_pose], dim=1)
    
            #fade in the feat from high resolution image
            skips_app = self.encoder_app(inp_app, fade_in_app, fade_in_alpha)
            #len(enc_filter), enc_c, h, w
    
            inp_pose = torch.cat([tg_pose, tg_parsing], dim=1)
            
            #fade in the feat from high resolution image        
            skips_pose = self.encoder_pose(inp_pose, fade_in_pose, fade_in_alpha)
            #len(enc_filter), enc_c, h, w
    
            # define concatenate func
            skips = self.concatenate_skips(skips_app, skips_pose, warps, masks)
            out, feat = self.decoder(skips)
    
            # return out and skips for local generator
            return out, feat, skips
    
        def concatenate_skips(self, skips_app, skips_pose, warps, masks):
            skips = []
            for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
                if i < 4:
                    out = AffineTransformLayer(10 if self.warp_skip == 'mask' else 1, self.image_size, self.warp_skip)(sk_app, warps, masks)
                    out = torch.cat([out, sk_pose], dim=1)
                else:
                    out = torch.cat([sk_app, sk_pose], dim=1)
                skips.append(out)
            return skips
    
    
    class Local_Generator(nn.Module):
        def __init__(self, input_nc, pose_dim, image_size, nfilters_enc, nfilters_dec, warp_skip, use_input_pose=True):
            super(Local_Generator, self).__init__()
            self.input_nc = input_nc
            # number of skip connections
            self.num_skips = 1 if warp_skip=='None' else 2
            self.warp_skip = warp_skip
            self.pose_dim = pose_dim
            self.nfilters_dec = nfilters_dec
            self.nfilters_enc = nfilters_enc
            self.image_size = image_size
            self.use_input_pose = use_input_pose
            self.pose_dim = 18
    
            # build global_generator
    
            ###### global generator model #####    
            self.model_global= Global_Generator(self.input_nc, self.pose_dim, (128,128), self.nfilters_enc, self.nfilters_dec, self.warp_skip, self.use_input_pose)
            self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    
    
            # local enhance layers
            self.model_local_encoder_app = local_encoder(input_nc-self.pose_dim - 9, nfilters_enc[:2])
            self.model_local_encoder_pose = local_encoder(self.pose_dim + 9, nfilters_enc[:2])
            self.model_local_decoder = local_decoder(nfilters_dec[-2:], nfilters_enc, self.num_skips)
            
    
        def get_imgpose(self, input, use_input_pose, pose_dim):
            inp_img = input[:, :12] # include pose and parsing
            inp_pose = input[:, 12:12 + pose_dim] if use_input_pose else None
    
            tg_parsing = input[:, 12+pose_dim: 21+pose_dim] # target parsing
            tg_pose_index = 21 + pose_dim if use_input_pose else 6
            tg_pose = input[:, tg_pose_index:]
            
            return inp_img, inp_pose, tg_parsing, tg_pose
    
        def forward(self, input, down_input, warps, masks, warps_128, masks_128,fade_in_alpha):
            
            inp_app, inp_pose, tg_parsing, tg_pose = self.get_imgpose(input, self.use_input_pose, self.pose_dim)
      
            inp_app = torch.cat([inp_app, inp_pose], dim=1)
            local_skips_app = self.model_local_encoder_app(inp_app)
            #skips_app:[32 x 256 x 256, 64 x 128 x 128]
    
     
            inp_pose = torch.cat([tg_pose, tg_parsing], dim=1)
            local_skips_pose = self.model_local_encoder_pose(inp_pose)
            #skips_pose: [32 x 256 x 256, 64 x 128 x 128]
    
            # define concatenate func
            local_skips = self.concatenate_skips(local_skips_app, local_skips_pose, warps, masks)
            # local_skips: [(32 + 32) x 256 x 256, (64 + 64) x 128 x 128]
    
            # downsample input to feed global_generator
            global_output, global_feat, global_skips = self.model_global(down_input, warps_128, masks_128, local_skips_app[1], local_skips_pose[1], fade_in_alpha) 
            # 3 x 256 x 256, 128 x 128 x 128, [(64 + 64) x 128 x 128, ...]
    
            # Concate the output of global skips and global output
            local_skips[1] = torch.cat([global_feat,global_skips[0]], dim=1)
            #local_skips: [(32 + 32) x 256 x 256, (128 + 64 + 64) x 128 x 128]
    
            out = self.model_local_decoder(local_skips)
    
            out = fade_in_alpha * out + (1-fade_in_alpha) * global_output
            return out
    
        def concatenate_skips(self, skips_app, skips_pose, warps, masks):
            skips = []
            for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
                out = AffineTransformLayer(10 if self.warp_skip == 'mask' else 1, self.image_size, self.warp_skip)(sk_app, warps, masks)
                out = torch.cat([out, sk_pose], dim=1)
               
                skips.append(out)
            return skips
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

Backward Process:

```
    	 def optimize_parameters(self):
            # forward
            self.forward()
    
            # freeze the pose detector
            self.set_requires_grad([self.netpose_det], False)
            self.set_requires_grad([self.downsample], False)
    
            # G
            self.set_requires_grad([self.netD, self.netD_face], False)
            self.optimizer_G.zero_grad()
            torch.nn.utils.clip_grad_norm(self.netG.parameters(), 100)
            self.backward_G()
            self.optimizer_G.step()
    
            # D
            self.set_requires_grad([self.netD, self.netD_face], True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
    
    		def backward_pose_det(self):
              
            self.loss_pose_det_A = self.criterionPose(self.fake_A_pose, self.pose_A) * 700 
            self.loss_pose_det_B = self.criterionPose(self.fake_B_pose, self.pose_B) * 700
            loss_pose_det = self.loss_pose_det_B + self.loss_pose_det_A
            return loss_pose_det
            
    
        def backward_D_basic(self, netD, real_img, fake_img, real_parsing, fake_parsing):
            # Real
            real = torch.cat((real_img, real_parsing), 1)
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            
            # Fake
            fake = torch.cat((fake_img, fake_parsing), 1)
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
    
            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            return loss_D
    
        def backward_D_face(self, netD, real_img, fake_img):
            # Real
            pred_real = netD(real_img)
            loss_D_real = self.criterionGAN(pred_real, True)
            
            # Fake
            pred_fake = netD(fake_img.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
    
            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            return loss_D
    
        def backward_D(self):
    
            # Train the general discriminator
            self.loss_D_A = self.backward_D_basic(self.netD, self.real_A, self.fake_A, self.A_parsing, self.A_parsing)
            self.loss_D_B = self.backward_D_basic(self.netD, self.real_A, self.fake_B, self.A_parsing, self.B_parsing)
    
             # Train the face discriminator
            self.loss_D_A_face = self.backward_D_face(self.netD_face, self.real_A_face, self.fake_A_face)
            self.loss_D_B_face = self.backward_D_face(self.netD_face, self.real_A_face, self.fake_B_face)
    
      
        def backward_G(self):
    
            self.D_fake_B = torch.cat((self.fake_B, self.B_parsing), 1)
            self.D_fake_A = torch.cat((self.fake_A, self.A_parsing), 1)
    
            # Train the general discriminator, as well as the face discriminator
            self.loss_G_A = self.criterionGAN(self.netD(self.D_fake_B), True) + self.criterionGAN(self.netD_face(self.fake_B_face), True)
            self.loss_G_B = self.criterionGAN(self.netD(self.D_fake_A), True) + self.criterionGAN(self.netD_face(self.fake_A_face), True)
    
    
            # pose consistency loss
            self.loss_pose_det = self.backward_pose_det()
    
            # Using content loss (L2)
            self.loss_content_loss = 0.03 * self.criterionIdt(self.fake_A_feat, self.real_A_feat)
    
            # semantic-aware loss
            self.loss_patch_style_real_A_fake_B = self.criterionSty(self.visibility * patch_gram_matrix(self.fake_B_feat, self.downsample_AtoB_masks), 
                                                                    self.visibility * patch_gram_matrix(self.real_A_feat, self.downsample_BtoA_masks))
            self.loss_patch_style_fake_A_fake_B = (self.visibility * patch_gram_matrix(self.fake_A_feat, self.downsample_BtoA_masks) - 
                                                   self.visibility * patch_gram_matrix(self.fake_B_feat, self.downsample_AtoB_masks)) ** 2
            self.loss_patch_style_fake_A_fake_B = self.loss_patch_style_fake_A_fake_B.mean()
    
            
            self.loss_patch_style = self.loss_patch_style_fake_A_fake_B + self.loss_patch_style_real_A_fake_B 
    
            self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_pose_det +  self.loss_content_loss + self.loss_patch_style
    
            self.loss_G.backward()
            
            
```