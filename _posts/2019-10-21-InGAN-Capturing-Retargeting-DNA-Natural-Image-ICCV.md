---
title: "InGAN: Capturing and Remapping the DNA of a Natural Image - Shocher - ICCV 2019"
tag:
- Image Synthesis
- Zero-Shot
---

## Info

- Title: InGAN: Capturing and Remapping the "DNA" of a Natural Image
- Task: Image Genration
- Author: Assaf Shocher, Shai Bagon, Phillip Isola, Michal Irani
- Date: Dec. 2018
- Arxiv: [1812.00231](https://arxiv.org/abs/1812.00231)
- Published: ICCV 2019



## Abstract

Generative Adversarial Networks (GANs) typically learn a distribution of images in a large image dataset, and are then able to generate new images from this distribution. However, each natural image has its own internal statistics, captured by its unique distribution of patches. In this paper we propose an "Internal GAN" (InGAN) - an image-specific GAN - which trains on a single input image and learns its internal distribution of patches. It is then able to synthesize a plethora of new natural images of significantly different sizes, shapes and aspect-ratios - all with the same internal patch-distribution (same "DNA") as the input image. In particular, despite large changes in global size/shape of the image, all elements inside the image maintain their local size/shape. InGAN is fully unsupervised, requiring no additional data other than the input image itself. Once trained on the input image, it can remap the input to any size or shape in a single feedforward pass, while preserving the same internal patch distribution. InGAN provides a unified framework for a variety of tasks, bridging the gap between textures and natural images.


![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](http://www.wisdom.weizmann.ac.il/~vision/ingan/resources/input_images/fruits.gif)


## Motivation & Design

### Overview

![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](https://i.imgur.com/BhyGuLC.png)

InGAN consists of a Generator G that retargets input x to output y whose size/shape is determined by a geometric transformation T (top left). A multiscale discriminator D learns to discriminate the patch statistics of the fake output y from the true patch statistics of the input image (right). Additionally, we take advantage of G’s automorphism to reconstruct the input back from y using G and the inverse transformation T −1 (bottom left).

The formulation aims to achieve two properties:

1. matching distributions: The distribution of patches, across scales, in the synthesized image, should match that distribution in the original input image. This property is a generalization of both the Coherence and Completeness objectives. 

2. localization: The elements’ locations in the generated image should generally match their relative locations in the original input image.



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




### Shape-flexible Generator

![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](https://i.imgur.com/FrsxWLG.png)

The desired geometric transformation for the output shape T is treated as an additional input that is fed to G for every forward pass. A parameter-free transformation layer geometrically transforms the feature map to the desired output shape. Making the transformation layer
parameter-free allows training G once to transform x to any size, shape or aspect ratio at test time.


### Multi-scale Patch Discriminator

![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](https://i.imgur.com/YgWSLpD.png)

InGAN uses a multi-scale D. This feature is significant: A single scale discriminator can only capture patch statistics of a specific size. Using a multiscale D matches the patch distribution over a range of patch sizes, capturing both fine-grained details as well as coarse structures in the image. At each scale, the discriminator is rather simple: it consists of just four conv-layers with the first one strided. Weights are not shared between different scale discriminators.




## Experiments & Ablation Study


Applications on image retargetting and texture synthesis can be found at the project page.





## Code

[Project Site](http://www.wisdom.weizmann.ac.il/~vision/ingan/index.html)

[PyTorch](https://github.com/assafshocher/InGAN)




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


### Model

```python
class InGAN:
    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf
        self.cur_iter = 0
        self.max_iters = conf.max_iters

        # Define input tensor
        self.input_tensor = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()
        self.real_example = torch.FloatTensor(1, 3, conf.output_crop_size, conf.output_crop_size).cuda()

        # Define networks
        self.G = networks.Generator(conf.G_base_channels, conf.G_num_resblocks, conf.G_num_downscales, conf.G_use_bias,
                                    conf.G_skip)
        self.D = networks.MultiScaleDiscriminator(conf.output_crop_size,  self.conf.D_max_num_scales,
                                                  self.conf.D_scale_factor, self.conf.D_base_channels)
        self.GAN_loss_layer = networks.GANLoss()
        self.Reconstruct_loss = networks.WeightedMSELoss(use_L1=conf.use_L1)
        self.RandCrop = networks.RandomCrop([conf.input_crop_size, conf.input_crop_size], must_divide=conf.must_divide)
        self.SwapCrops = networks.SwapCrops(conf.crop_swap_min_size, conf.crop_swap_max_size)

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward
        self.criterionReconstruction = self.Reconstruct_loss.forward

        # Keeping track of losses- prepare tensors
        self.losses_G_gan = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_D_real = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_D_fake = torch.FloatTensor(conf.print_freq).cuda()
        self.losses_G_reconstruct = torch.FloatTensor(conf.print_freq).cuda()
        if self.conf.reconstruct_loss_stop_iter > 0:
            self.losses_D_reconstruct = torch.FloatTensor(conf.print_freq).cuda()

        # Initialize networks
        self.G.apply(networks.weights_init)
        self.D.apply(networks.weights_init)

        # Initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))

        # Learning rate scheduler
        # First define linearly decaying functions (decay starts at a special iter)
        start_decay = conf.lr_start_decay_iter
        end_decay = conf.max_iters
        # def lr_function(n_iter):
        #     return 1 - max(0, 1.0 * (n_iter - start_decay) / (conf.max_iters - start_decay))
        lr_function = LRPolicy(start_decay, end_decay)
        # Define learning rate schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_function)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_function)

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        # Determine output size of G (dynamic change)
        output_size, random_affine = random_size(orig_size=self.input_tensor.shape[2:],
curriculum=self.conf.curriculum,
i=self.cur_iter,
iter_for_max_range=self.conf.iter_for_max_range,
must_divide=self.conf.must_divide,
min_scale=self.conf.min_scale,
max_scale=self.conf.max_scale,
max_transform_magniutude=self.conf.max_transform_magnitude)

        # Add noise to G input for better generalization (make it ignore the 1/255 binning)
        self.input_tensor_noised = self.input_tensor + (torch.rand_like(self.input_tensor) - 0.5) * 2.0 / 255

        # Generator forward pass
        self.G_pred = self.G.forward(self.input_tensor_noised, output_size=output_size, random_affine=random_affine)

        # Run generator result through discriminator forward pass
        self.scale_weights = get_scale_weights(i=self.cur_iter,
max_i=self.conf.D_scale_weights_iter_for_even_scales,
start_factor=self.conf.D_scale_weights_sigma,
input_shape=self.G_pred.shape[2:],
min_size=self.conf.D_min_input_size,
num_scales_limit=self.conf.D_max_num_scales,
scale_factor=self.conf.D_scale_factor)
        d_pred_fake = self.D.forward(self.G_pred, self.scale_weights)

        # If reconstruction-loss is used, run through decoder to reconstruct, then calculate reconstruction loss
        if self.conf.reconstruct_loss_stop_iter > self.cur_iter:
            self.reconstruct = self.G.forward(self.G_pred, output_size=self.input_tensor.shape[2:], random_affine=-random_affine)
            self.loss_G_reconstruct = self.criterionReconstruction(self.reconstruct, self.input_tensor, self.loss_mask)

        # Calculate generator loss, based on discriminator prediction on generator result
        self.loss_G_GAN = self.criterionGAN(d_pred_fake, is_d_input_real=True)

        # Generator final loss
        # Weighted average of the two losses (if indicated to use reconstruction loss)
        if self.conf.reconstruct_loss_stop_iter < self.cur_iter:
            self.loss_G = self.loss_G_GAN
        else:
            self.loss_G = (self.conf.reconstruct_loss_proportion * self.loss_G_reconstruct + self.loss_G_GAN)

        # Calculate gradients
        # Note that the gradients are propagated from the loss through discriminator and then through generator
        self.loss_G.backward()

        # Update weights
        # Note that only generator weights are updated (by definition of the G optimizer)
        self.optimizer_G.step()

        # Extra training for the inverse G. The difference between this and the reconstruction is the .detach() which
        # makes the training only for the inverse G and not for regular G.
        if self.cur_iter > self.conf.G_extra_inverse_train_start_iter:
            for _ in range(self.conf.G_extra_inverse_train):
                self.optimizer_G.zero_grad()
                self.inverse = self.G.forward(self.G_pred.detach(), output_size=self.input_tensor.shape[2:], random_affine=-random_affine)
                self.loss_G_inverse = (self.criterionReconstruction(self.inverse, self.input_tensor, self.loss_mask) *
                                       self.conf.G_extra_inverse_train_ratio)
                self.loss_G_inverse.backward()
                self.optimizer_G.step()

        # Update learning rate scheduler
        self.lr_scheduler_G.step()

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()

        # Adding noise to D input to prevent overfitting to 1/255 bins
        real_example_with_noise = self.real_example + (torch.rand_like(self.real_example[-1]) - 0.5) * 2.0 / 255.0

        # Discriminator forward pass over real example
        self.d_pred_real = self.D.forward(real_example_with_noise, self.scale_weights)

        # Adding noise to D input to prevent overfitting to 1/255 bins
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_pred_with_noise = self.G_pred.detach() + (torch.rand_like(self.G_pred) - 0.5) * 2.0 / 255

        # Discriminator forward pass over generated example example
        self.d_pred_fake = self.D.forward(g_pred_with_noise, self.scale_weights)

        # Calculate discriminator loss
        self.loss_D_fake = self.criterionGAN(self.d_pred_fake, is_d_input_real=False)
        self.loss_D_real = self.criterionGAN(self.d_pred_real, is_d_input_real=True)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        # Calculate gradients
        # Note that gradients are not propagating back through generator
        # noinspection PyUnresolvedReferences
        self.loss_D.backward()

        # Update weights
        # Note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

        # Update learning rate scheduler
        self.lr_scheduler_D.step()

    def train_one_iter(self, cur_iter, input_tensors):
        # Set inputs as random crops
        input_crops = []
        mask_crops = []
        real_example_crops = []
        mask_flag = False
        for input_tensor in input_tensors:
            real_example_crops += self.RandCrop.forward([input_tensor])

            if np.random.rand() < self.conf.crop_swap_probability:
                swapped_input_tensor, loss_mask = self.SwapCrops.forward(input_tensor)
                [input_crop, mask_crop] = self.RandCrop.forward([swapped_input_tensor, loss_mask])
                input_crops.append(input_crop)
                mask_crops.append(mask_crop)
                mask_flag = True
            else:
                input_crops.append(real_example_crops[-1])

        self.input_tensor = torch.cat(input_crops)
        self.real_example = torch.cat(real_example_crops)
        self.loss_mask = torch.cat(mask_crops) if mask_flag else None

        # Update current iteration
        self.cur_iter = cur_iter

        # Run a single forward-backward pass on the model and update weights
        # One global iteration includes several iterations of generator and several of discriminator
        # (not necessarily equal)
        # noinspection PyRedeclaration
        for _ in range(self.conf.G_iters):
            self.train_g()

        # noinspection PyRedeclaration
        for _ in range(self.conf.D_iters):
            self.train_d()

        # Accumulate stats
        # Accumulating as cuda tensors is much more efficient than passing info from GPU to CPU at every iteration
        self.losses_G_gan[cur_iter % self.conf.print_freq] = self.loss_G_GAN.item()
        self.losses_D_fake[cur_iter % self.conf.print_freq] = self.loss_D_fake.item()
        self.losses_D_real[cur_iter % self.conf.print_freq] = self.loss_D_real.item()
        if self.conf.reconstruct_loss_stop_iter > self.cur_iter:
            self.losses_G_reconstruct[cur_iter % self.conf.print_freq] = self.loss_G_reconstruct.item()
```


### Generator

![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](https://i.imgur.com/FrsxWLG.png)


```python
class Generator(nn.Module):
    """ Architecture of the Generator, uses res-blocks """

    def __init__(self, base_channels=64, n_blocks=6, n_downsampling=3, use_bias=True, skip_flag=True):
        super(Generator, self).__init__()

        # Determine whether to use skip connections
        self.skip = skip_flag

        # Entry block
        # First conv-block, no stride so image dims are kept and channels dim is expanded (pad-conv-norm-relu)
        self.entry_block = nn.Sequential(nn.ReflectionPad2d(3),
nn.utils.spectral_norm(nn.Conv2d(3, base_channels, kernel_size=7, bias=use_bias)),normalization_layer(base_channels),nn.LeakyReLU(0.2, True))

        # Geometric transformation
        self.geo_transform = GeoTransform()

        # Downscaling
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        self.downscale_block = RescaleBlock(n_downsampling, 0.5, base_channels, True)

        # Bottleneck
        # A sequence of res-blocks
        bottleneck_block = []
        for _ in range(n_blocks):
            # noinspection PyUnboundLocalVariable
            bottleneck_block += [ResnetBlock(base_channels * 2 ** n_downsampling, use_bias=use_bias)]
        self.bottleneck_block = nn.Sequential(*bottleneck_block)

        # Upscaling
        # A sequence of transposed-conv-blocks, Image dims expand by 2, channels dim shrinks by 2 at each block\
        self.upscale_block = RescaleBlock(n_downsampling, 2.0, base_channels, True)

        # Final block
        # No stride so image dims are kept and channels dim shrinks to 3 (output image channels)
        self.final_block = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(base_channels, 3, kernel_size=7),
                                         nn.Tanh())

    def forward(self, input_tensor, output_size, random_affine):
        # A condition for having the output at same size as the scaled input is having even output_size

        # Entry block
        feature_map = self.entry_block(input_tensor)

        # Change scale to output scale by interpolation
        if random_affine is None:
            feature_map = f.interpolate(feature_map, size=output_size, mode='bilinear')
        else:
            feature_map = self.geo_transform.forward(feature_map, output_size, random_affine)

        # Downscale block
        feature_map, downscales = self.downscale_block.forward(feature_map, return_all_scales=self.skip)

        # Bottleneck (res-blocks)
        feature_map = self.bottleneck_block(feature_map)

        # Upscale block
        feature_map, _ = self.upscale_block.forward(feature_map, pyramid=downscales, skip=self.skip)

        # Final block
        output_tensor = self.final_block(feature_map)

        return output_tensor

```


### Discriminator


![InGAN: Capturing and Remapping the "DNA" of a Natural Image - Shocher - ICCV 2019](https://i.imgur.com/YgWSLpD.png)

```python

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, real_crop_size, max_n_scales=9, scale_factor=2, base_channels=128, extra_conv_layers=0):
        super(MultiScaleDiscriminator, self).__init__()
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.min_size = 16
        self.extra_conv_layers = extra_conv_layers

        # We want the max num of scales to fit the size of the real examples. further scaling would create networks that
        # only train on fake examples
        self.max_n_scales = np.min([np.int(np.ceil(np.log(np.min(real_crop_size) * 1.0 / self.min_size)
                                                   / np.log(self.scale_factor))), max_n_scales])

        # Prepare a list of all the networks for all the wanted scales
        self.nets = nn.ModuleList()

        # Create a network for each scale
        for _ in range(self.max_n_scales):
            self.nets.append(self.make_net())

    def make_net(self):
        base_channels = self.base_channels
        net = []

        # Entry block
        net += [nn.utils.spectral_norm(nn.Conv2d(3, base_channels, kernel_size=3, stride=1)),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, True)]

        # Downscaling blocks
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        net += [nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Regular conv-block
        net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
out_channels=base_channels * 2, kernel_size=3, bias=True)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Additional 1x1 conv-blocks
        for _ in range(self.extra_conv_layers):
            net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,out_channels=base_channels * 2,kernel_size=3,bias=True)),
                    nn.BatchNorm2d(base_channels * 2),
                    nn.LeakyReLU(0.2, True)]

        # Final conv-block
        # Ends with a Sigmoid to get a range of 0-1
        net += nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, 1, kernel_size=1)),nn.Sigmoid())

        # Make it a valid layers sequence and return
        return nn.Sequential(*net)

    def forward(self, input_tensor, scale_weights):
        aggregated_result_maps_from_all_scales = self.nets[0](input_tensor) * scale_weights[0]
        map_size = aggregated_result_maps_from_all_scales.shape[2:]

        # Run all nets over all scales and aggregate the interpolated results
        for net, scale_weight, i in zip(self.nets[1:], scale_weights[1:], range(1, len(scale_weights))):
            downscaled_image = f.interpolate(input_tensor, scale_factor=self.scale_factor**(-i), mode='bilinear')
            result_map_for_current_scale = net(downscaled_image)
            upscaled_result_map_for_current_scale = f.interpolate(result_map_for_current_scale,
                                                                  size=map_size,
                                                                  mode='bilinear')
            aggregated_result_maps_from_all_scales += upscaled_result_map_for_current_scale * scale_weight

        return aggregated_result_maps_from_all_scales
```

## Related

- [ICCV 2019: Image Synthesis(Part One)](https://arxivnote.ddlee.cn/2019/10/30/Image-Synthesis-Generation-ICCV-2019.html)
- [ICCV 2019: Image Synthesis(Part Two)](https://arxivnote.ddlee.cn/2019/10/30/Image-Synthesis-Generation-ICCV-2019-2.html)
- [ICCV 2019: Image-to-Image Translation](https://arxivnote.ddlee.cn/2019/10/24/Image-to-Image-Translation-ICCV-2019.html)
- [GANs for Image Generation: ProGAN, SAGAN, BigGAN, StyleGAN](https://cvnote.ddlee.cn/2019/09/15/ProGAN-SAGAN-BigGAN-StyleGAN.html)
- [GANs in PyTorch: DCGAN, cGAN, LSGAN, InfoGAN, WGAN and more](https://cvnote.ddlee.cn/2019/09/25/GANs-PyTorch.html)
- [Common Training Loss Curve of DCGAN and WGAN](https://cvnote.ddlee.cn/2019/09/26/DCGAN-WGAN-Training-Loss-Curve.html)

