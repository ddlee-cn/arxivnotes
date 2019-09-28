---
title: Progressive Pose Attention Transfer for Person Image Generation - Zhen Zhu - CVPR 2019
tag:
- Image Generation
- Application
- GAN
- Attention
---



## Info

- Title: Progressive Pose Attention Transfer for Person Image Generation
- Task: Image Generation
- Author: Zhen Zhu , Tengteng Huang , Baoguang Shi, Miao Yu , Bofei Wang , Xiang Bai
- Arxiv: [1904.03349](https://arxiv.org/abs/1904.03349)
- Published: CVPR 2019

## Highlights

- We propose a progressive pose attention transfer network to address the challenging task of pose transfer, which is neat in design and efficient in computation.

- The proposed network leverages a novel cascaded Pose-Attentional Transfer Blocks (PATBs) that can effectively utilize pose and appearance features to smoothly guide the pose transfer process.

## Abstract

This paper proposes a new generative adversarial network for pose transfer, i.e., transferring the pose of a given person to a target pose. The generator of the network comprises a sequence of Pose-Attentional Transfer Blocks that each transfers certain regions it attends to, generating the person image progressively. Compared with those in previous works, our generated person images possess better appearance consistency and shape consistency with the input images, thus significantly more realistic-looking. The efficacy and efficiency of the proposed network are validated both qualitatively and quantitatively on Market-1501 and DeepFashion. Furthermore, the proposed architecture can generate training images for person re-identification, alleviating data insufficiency. Codes and models are available at: [https://github.com/tengteng95/Pose-Transfer.git](https://github.com/tengteng95/Pose-Transfer.git).

## Motivation & Design

![Progressive Pose Attention Transfer for Person Image Generation - Zhen Zhu - CVPR 2019](https://i.imgur.com/zzVHnnv.png)



## Experiments & Ablation Study

The progressive behavior of the proposed model, visualized by the attention map in PATB module.

![Progressive Pose Attention Transfer for Person Image Generation - Zhen Zhu - CVPR 2019](https://i.imgur.com/kUoitFR.png)

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

Application: data augmentation for Person Re-ID
![Progressive Pose Attention Transfer for Person Image Generation - Zhen Zhu - CVPR 2019](https://i.imgur.com/Y4QuLOb.png)

## Code

[PyTorch]([https://github.com/tengteng95/Pose-Transfer](https://github.com/tengteng95/Pose-Transfer))

```
class PATBlock(nn.Module):
        def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False):
            super(PATBlock, self).__init__()
            self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
            self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=True, cated_stream2=cated_stream2)
    
        def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False, cal_att=False):
            conv_block = []
            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    
            if cated_stream2:
                conv_block += [nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim*2),
                           nn.ReLU(True)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                               norm_layer(dim),
                               nn.ReLU(True)]
            if use_dropout:
                conv_block += [nn.Dropout(0.5)]
    
            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    
            if cal_att:
                if cated_stream2:
                    conv_block += [nn.Conv2d(dim*2, dim, kernel_size=3, padding=p, bias=use_bias)]
                else:
                    conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim)]
    
            return nn.Sequential(*conv_block)
    
        def forward(self, x1, x2):
            x1_out = self.conv_block_stream1(x1)
            x2_out = self.conv_block_stream2(x2)
            att = F.sigmoid(x2_out)
    
            x1_out = x1_out * att
            out = x1 + x1_out # residual connection
    
            # stream2 receive feedback from stream1
            x2_out = torch.cat((x2_out, out), 1)
            return out, x2_out, x1_out

    class PATNModel(nn.Module):
        def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
            assert(n_blocks >= 0 and type(input_nc) == list)
            super(PATNModel, self).__init__()
            self.input_nc_s1 = input_nc[0]
            self.input_nc_s2 = input_nc[1]
            self.output_nc = output_nc
            self.ngf = ngf
            self.gpu_ids = gpu_ids
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d
    
            # down_sample
            model_stream1_down = [nn.ReflectionPad2d(3),
                        nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0,
                               bias=use_bias),
                        norm_layer(ngf),
                        nn.ReLU(True)]
    
            model_stream2_down = [nn.ReflectionPad2d(3),
                        nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0,
                               bias=use_bias),
                        norm_layer(ngf),
                        nn.ReLU(True)]
    
            # n_downsampling = 2
            for i in range(n_downsampling):
                mult = 2**i
                model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                                norm_layer(ngf * mult * 2),
                                nn.ReLU(True)]
                model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                                norm_layer(ngf * mult * 2),
                                nn.ReLU(True)]
    
            # att_block in place of res_block
            mult = 2**n_downsampling
            cated_stream2 = [True for i in range(n_blocks)]
            cated_stream2[0] = False
            attBlock = nn.ModuleList()
            for i in range(n_blocks):
                attBlock.append(PATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cated_stream2=cated_stream2[i]))
    
            # up_sample
            model_stream1_up = []
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                                norm_layer(int(ngf * mult / 2)),
                                nn.ReLU(True)]
    
            model_stream1_up += [nn.ReflectionPad2d(3)]
            model_stream1_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model_stream1_up += [nn.Tanh()]
    
            # self.model = nn.Sequential(*model)
            self.stream1_down = nn.Sequential(*model_stream1_down)
            self.stream2_down = nn.Sequential(*model_stream2_down)
            # self.att = nn.Sequential(*attBlock)
            self.att = attBlock
            self.stream1_up = nn.Sequential(*model_stream1_up)
    
        def forward(self, input): # x from stream 1 and stream 2
            # here x should be a tuple
            x1, x2 = input
            # down_sample
            x1 = self.stream1_down(x1)
            x2 = self.stream2_down(x2)
            # att_block
            for model in self.att:
                x1, x2, _ = model(x1, x2)
    
            # up_sample
            x1 = self.stream1_up(x1)
    
            return x1
            
```

    ## Related
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
