---
title: Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation - ICCV 2019 - PyTorch
tag:
- Image-to-Image Translation
- GAN
---

## Info

- Title:  Interactive Sketch & Fill: Multiclass Sketch-to-Image Translation
- Task: Image-to-Image Translation
- Author:
- Date: Sep. 2019
- Arxiv: [1909.11081](https://arxiv.org/abs/1909.11081)
- Published: ICCV 2019



## Highlights

- Interactive GUI
- First complete, then generate
- Gating-based approach for class conditioning

## Abstract

We propose an interactive GAN-based sketch-to-image translation method that helps novice users create images of simple objects. As the user starts to draw a sketch of a desired object type, the network interactively recommends plausible completions, and shows a corresponding synthesized image to the user. This enables a feedback loop, where the user can edit their sketch based on the network's recommendations, visualizing both the completed shape and final rendered image while they draw. In order to use a single trained model across a wide array of object classes, we introduce a gating-based approach for class conditioning, which allows us to generate distinct classes without feature mixing, from a single generator network. 



## Motivation & Design



### 2-Stage Manner

The model first completes the user input and then generates an image conditioned on the completed shape. There are several advantages to  this two-stage approach. For one, we are able to give the
artist feedback on the general object shape in our interactive interface , allowing them to quickly refine higher level shape until it is satisfactory. Second, we found that splitting completion and image generation to work better than going directly from partial outlines to images, as the additional intermediate supervision on full outlines/sketches breaks the problem into two easier sub-problems – first recover the geometric properties of the object (shape, proportions) and then fill in the appearance(colors, textures).




![](https://arnabgho.github.io/iSketchNFill/resources/imgs/method.png)

First, complete a partial sketch using the shape generator $G_S$ . Then translate the completed sketch into an image using the appearance generator $G_A$ . Both generators are trained with their respective discriminators $D_S$ , and $D_A$.





### Stage 1: Sketch Completion

To achieve multi-modal completions, the shape generator is designed using inspiration from non-image conditional model with the conditioning input provided at multiple scales, so that the generator network doesn't ignore the partial stroke conditioning.



![](https://i.imgur.com/9gEjSrP.png)



### Stage 2: Sketch-to-Image Translation

For the second stage, we use a multi-class generator that is conditioned on a user supplied class label. This generator applies a gating mechanism that allows the network to focus on the important parts (activations) of the network specific to a given class. Such an approach allows for a clean separation of classes, enabling us to train a single generator and discriminator across multiple object classes, therebyenabling a finite-size deployable model that can be used in multiple different scenarios.




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




### Gating Mechanism



![](https://i.imgur.com/FJ2P4KO.png)



The model uses gating on all the residual blocks of the generator and the discriminator, other forms of conditioning such as (naive concatenation in input only, all layers, AC-GAN like latent regressor are evaluated as well.



![](https://i.imgur.com/PZIFGrI.png)



(Left) A “vanilla” residual block without conditioning applies a residual modification to the input tensor. 

(Mid-left) The $H(X)$ block is softly-gated by scalar parameter α and shift $β$. 

(Mid) Adaptive Instance Normalization applies a channel-wise scaling and shifting after an instance normalization layer. 

(Mid-right) Channel-wise gating adds restrictions Class to label the range injection of $α$. 

(Right) We find that channel-wise gating (without added bias) produces the best results empirically.





## Experiments & Ablation Study


![](https://i.imgur.com/OBtStTn.png)



(Top) Given a user created incomplete object outline (first row), our model estimates the complete shape and provides this as a recommendation to the user (shown in gray), along with the final synthesized object (second row). These estimates are updated as the user adds (green) or removes (red) strokes over time – previous edits are shown in black.

(Bottom) This generation is class-conditioned, and our method is able to generate distinct multiple objects for the same outline (e.g. ‘circle’) by conditioning the generator on the object category.




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


![](https://i.imgur.com/B3pIDrR.png)

**Learned gating parameters**
We show the soft-gating parameters for (left) blockwise and (right) channelwise
gating for the (top) generator and (bot) discriminator. Black indicates completely off, and white indicates completely on. For channelwise, a subset (every 4th) of blocks is shown. Within each block, channels are sorted in ascending order of the first category. The nonuniformity of each columns indicates that different channels are used more heavily for different classes.



## Code

[Project Site](https://arnabgho.github.io/iSketchNFill/)

[PyTorch](https://github.com/arnabgho/iSketchNFill)

### Stage 1: Sparse WGAN-GP Pix2Pix Model

```python
class SparseWGANGPPix2PixModel(BaseModel):
    def name(self):
        return 'SparseWGANGPPix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.sparse_input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.sparseSize, opt.sparseSize)
        self.mask_input_A = self.Tensor(opt.batchSize, 1,
                                   opt.fineSize, opt.fineSize)


        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.label = self.Tensor(opt.batchSize,1)
        if opt.nz>0:
            self.noise=self.Tensor(opt.batchSize,opt.nz)
            self.test_noise= self.get_z_random(opt.num_interpolate,opt.nz)
            self.test_noise.normal_(0,0.2)
        # load/define networks
        opt.which_model_netG = 'GAN_stability_Generator'


        self.netG = networks_sparse.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,opt)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            opt.which_model_netD = 'GAN_stability_Discriminator'
            self.netD = networks_sparse.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,opt)

            if self.isTrain:
                self.netD = nn.DataParallel(self.netD)
                self.netD.to(device)
            else:
                self.netD.cuda()

        if self.isTrain:
            self.netG = nn.DataParallel(self.netG)
            self.netG.to(device)
        else:
            self.netG.cuda()

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.WGANLoss(tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr_g, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr_d, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)

	def forward(self):
        self.sparse_real_A = Variable(self.input_A)
        if self.opt.nz>0:
            self.fake_B = self.netG(self.real_A,self.label,self.noise)
        else:
            self.fake_B = self.netG(self.real_A,self.label)
        self.real_B = Variable(self.input_B)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.opt.img_conditional_D:
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB.detach(),self.label)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        if self.opt.img_conditional_D:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            real_AB = self.real_B

        pred_real = self.netD(real_AB,self.label)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()
        self.reg = self.opt.wgan_gp_lambda * self.wgan_gp_reg(real_AB,fake_AB,self.label,center= self.opt.wgan_gp_center)
        self.reg.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.img_conditional_D:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB,self.label)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

        # Second, G(A) = B
        mask_A_resized = self.mask_input_A.expand_as(self.fake_B)

        self.loss_G_L1 = self.criterionL1(self.fake_B*mask_A_resized, self.real_A) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.netD(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

```





### Stage 1: Sparse WGAN-GP Generator

```python
class GAN_stability_Generator(nn.Module):
    def __init__(self, opt , embed_size=256, nfilter=64, **kwargs):
        super().__init__()
        self.opt = opt
        size = opt.fineSize
        nlabels = opt.n_classes
        s0 = self.s0 = size // 32
        nf = self.nf = opt.ngf
        self.z_dim = z_dim = opt.nz
        nc = opt.input_nc
        # Submodules
        self.embedding = nn.Embedding(nlabels, embed_size)
        self.fc = nn.Linear(z_dim + embed_size, 16*nf*s0*s0)
        self.resnet_0_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_0_1 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_0 = ResnetBlock(16*nf, 16*nf)
        self.resnet_1_1 = ResnetBlock(16*nf, 16*nf)
        self.resnet_2_0 = ResnetBlock(16*nf, 8*nf)
        self.resnet_2_1 = ResnetBlock(8*nf, 8*nf)
        self.resnet_3_0 = ResnetBlock(8*nf, 4*nf)
        self.resnet_3_1 = ResnetBlock(4*nf, 4*nf)
        self.resnet_4_0 = ResnetBlock(4*nf, 2*nf)
        self.resnet_4_1 = ResnetBlock(2*nf, 2*nf)
        self.resnet_5_0 = ResnetBlock(2*nf, 1*nf)
        self.resnet_5_1 = ResnetBlock(1*nf, 1*nf)
        self.conv_img = nn.Conv2d(nf, opt.output_nc, 3, padding=1)

        sparse_processor_blocks = []
        # 8x8
        sparse_processor_blocks += [GatedResnetBlock(nc,16*nf)]
        # 16x16
        sparse_processor_blocks += [GatedResnetBlock(nc,16*nf)]
        # 32x32
        sparse_processor_blocks += [GatedResnetBlock(nc,8*nf)]
        # 64x64
        sparse_processor_blocks += [GatedResnetBlock(nc,4*nf)]
        # 128x128
        sparse_processor_blocks += [GatedResnetBlock(nc,2*nf)]
        
        self.num_sparse_blocks = len(sparse_processor_blocks)
        self.sparse_processor = nn.Sequential(*sparse_processor_blocks)


    def forward(self, sparse_input , y, z):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            yembed = self.embedding(y)
        else:
            yembed = y

        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        yz = torch.cat([z, yembed], dim=1)
        out = self.fc(yz)
        out = out.view(batch_size, 16*self.nf, self.s0, self.s0)

        scale_factor = 1.0/32.0

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[0](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[1](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[2](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[3](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)

        sparse = F.interpolate(sparse_input,scale_factor=scale_factor)
        sparse = self.sparse_processor[4](sparse)
        scale_factor *= 2.0
        out += sparse
        out = F.upsample(out, scale_factor=2)
        out = self.resnet_5_0(out)
        out = self.resnet_5_1(out)

        if self.opt.no_sparse_add:
            out = self.conv_img(actvn(out))
        else:
            out = sparse_input + self.conv_img(actvn(out))
        out = F.tanh(out)

        return out

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





### Stage 2: Channel-wise Gated Conditioning Generator



![](https://i.imgur.com/aYg8BZO.png)



```python
class StochasticLabelBetaChannelGatedResnetConvResnetG(nn.Module):
    def __init__(self,opt):
        super(StochasticLabelBetaChannelGatedResnetConvResnetG, self).__init__()
        self.opt=opt
        opt.nsalient = max(10,opt.n_classes)
        self.label_embedding = nn.Embedding(opt.n_classes, opt.nsalient)
        self.main_initial = nn.Sequential( nn.Conv2d(3,opt.ngf,kernel_size=3,stride=1,padding=1) ,
                            get_norm(opt.ngf,opt.norm_G,opt.num_groups),
                            nn.ReLU(True)
                            )
        self.label_noise = nn.Linear(opt.nz,opt.nsalient)
        main_block=[]
        #Input is z going to series of rsidual blocks

        # Sets of residual blocks start
        for i in range(3):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op)]


        for i in range(opt.ngres_up_down):
            main_block += [ DownGatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op) ]

        for i in range(int(opt.ngres/2-opt.ngres_up_down-3)):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op)]


        for i in range(int(opt.ngres/2-opt.ngres_up_down-3)):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op)]



        for i in range(opt.ngres_up_down):
            main_block += [ UpGatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G,num_groups=opt.num_groups,res_op=opt.res_op ) ]

        for i in range(3):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout_G,use_sn=opt.spectral_G,norm_layer=opt.norm_G , num_groups = opt.num_groups,res_op=opt.res_op )]



        # Final layer to map to 3 channel
        if opt.spectral_G:
            main_block+=[spectral_norm(nn.Conv2d(opt.ngf,opt.nc,kernel_size=3,stride=1,padding=1)) ]
        else:
            main_block+=[nn.Conv2d(opt.ngf,opt.nc,kernel_size=3,stride=1,padding=1) ]
        main_block+=[nn.Tanh()]
        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ Reshape( -1, 1 ,opt.nsalient)  ]
        gate_block+=[ nn.Conv1d(1,opt.ngf_gate,kernel_size=3,stride=1,padding=1)  ]
        gate_block+=[ nn.ReLU()]
        for i in range(opt.ngres_gate):
            gate_block+=[ResBlock1D(opt.ngf_gate,opt.dropout_gate)]
        # state size (opt.batchSize, opt.ngf_gate, opt.nsalient)
        gate_block+=[Reshape(-1,opt.ngf_gate*opt.nsalient)]

        self.gate=nn.Sequential(*gate_block)

        gate_block_mult = []
        gate_block_mult+=[ nn.Linear(opt.ngf_gate*opt.nsalient,opt.ngres*opt.ngf) ]
        gate_block_mult+= [ nn.Sigmoid()]

        self.gate_mult = nn.Sequential(*gate_block_mult)


        gate_block_add = gate_block
        gate_block_add+=[ nn.Linear(opt.ngf_gate*opt.nsalient,opt.ngres*opt.ngf) ]
        gate_block_add+= [nn.Hardtanh()]
        self.gate_add = nn.Sequential(*gate_block_add)


    def forward(self, input, labels, noise=None):
        input_gate = self.label_embedding(labels)
        input_noise=self.label_noise(noise)

        # Things are just flipped here
        output_gate = self.gate(input_noise)
        output_gate_mult = self.gate_mult(output_gate)
        output_gate_add = self.gate_add(input_gate)
        output = self.main_initial(input)
        for i in range(self.opt.ngres):
            alpha = output_gate_mult[:,i*self.opt.ngf:(i+1)*self.opt.ngf]
            alpha = alpha.resize(self.opt.batchSize,self.opt.ngf,1,1)
            beta=output_gate_add[:,i*self.opt.ngf:(i+1)*self.opt.ngf]
            beta=beta.resize(self.opt.batchSize,self.opt.ngf,1,1)
            output=self.main[i](output,alpha,beta)

        output=self.main[self.opt.ngres](output)
        output=self.main[self.opt.ngres+1](output)
        return output

```



### Stage 2: Channel-wise Gated Conditioning Discriminator



![](https://i.imgur.com/jiAB2Jq.png)





```python

class LabelChannelGatedResnetConvResnetD(nn.Module):
    def __init__(self,opt,input_nc=6, ndf=32, n_layers=0, norm_layer=nn.BatchNorm2d, use_sigmoid=True, gpu_ids=[],use_sn=False):
        super(LabelChannelGatedResnetConvResnetD, self).__init__()
        self.opt=opt
        opt.nsalient = max(10,opt.n_classes)
        self.label_embedding = nn.Embedding(opt.n_classes, opt.nsalient)
        use_sn = opt.spectral_D
        use_sigmoid = opt.no_lsgan
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ndf= opt.ndf
        kw = 4
        padw = 1

        sequence = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            #nf_mult = min(2**n, 8)
            if use_sn:
                sequence += [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                    get_norm(ndf*nf_mult,opt.norm_D,opt.num_groups),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    get_norm(ndf*nf_mult,opt.norm_D,opt.num_groups),
                    nn.LeakyReLU(0.2, True)
                ]

        if use_sn:
            sequence += [spectral_norm( nn.Conv2d(ndf * nf_mult, opt.ndisc_out_filters, kernel_size=kw, stride=1, padding=padw) ) ]
        else:
            sequence += [ nn.Conv2d(ndf * nf_mult, opt.ndisc_out_filters, kernel_size=kw, stride=1, padding=padw) ]


        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.main_latter = nn.Sequential(*sequence)


        main_block=[]
        #Input is z going to series of rsidual blocks
        # First layer to map to ndf channel
        if opt.spectral_D:
            main_block+=[spectral_norm(nn.Conv2d(opt.input_nc + opt.output_nc,opt.ndf,kernel_size=3,stride=1,padding=1))]
        else:
            main_block+=[nn.Conv2d(opt.input_nc + opt.output_nc ,opt.ndf,kernel_size=3,stride=1,padding=1)]
        # Sets of residual blocks start

        for i in range(3):
            main_block+= [GatedConvResBlock(opt.ndf,opt.ndf,dropout=opt.dropout,use_sn=opt.spectral_D,norm_layer=opt.norm_D,num_groups=opt.num_groups,res_op=opt.res_op)]

        for i in range(opt.ndres_down):
            main_block+= [DownGatedConvResBlock(opt.ndf,opt.ndf,dropout=opt.dropout_D,use_sn=opt.spectral_D,norm_layer=opt.norm_D,num_groups=opt.num_groups,res_op=opt.res_op)]

        for i in range(opt.ndres - opt.ndres_down-3  ):
            main_block+= [GatedConvResBlock(opt.ndf,opt.ndf,dropout=opt.dropout_D,use_sn=opt.spectral_D,norm_layer=opt.norm_D , num_groups=opt.num_groups ,res_op=opt.res_op)]


        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ Reshape( -1, 1 ,opt.nsalient)  ]
        gate_block+=[ nn.Conv1d(1,opt.ngf_gate,kernel_size=3,stride=1,padding=1)  ]


        gate_block+=[ nn.ReLU()]
        for i in range(opt.ndres_gate):
            gate_block+=[ResBlock1D(opt.ndf_gate,opt.dropout_gate)]
        # state_size (opt.batchSize,opt.ndf_gate,opt.nsalient)
        gate_block+= [Reshape(-1,opt.ndf_gate*opt.nsalient)]

        self.gate = nn.Sequential(*gate_block)

        gate_block_mult=[]
        gate_block_mult+=[ nn.Linear(opt.ndf_gate*opt.nsalient,opt.ndres*opt.ndf) ]
        gate_block_mult+= [nn.Sigmoid()]

        self.gate_mult = nn.Sequential(*gate_block_mult)

        if opt.gate_affine:
            gate_block_add = []
            gate_block_add+=[ nn.Linear(opt.ndf_gate*opt.nsalient,opt.ndres*opt.ndf) ]
            gate_block_add+=[nn.Tanh()]
            self.gate_add=nn.Sequential(*gate_block_add)

    def forward(self, img, labels):
        batchSize=labels.size(0)
        input_gate = self.label_embedding(labels)
        input_main = img

        output_gate = self.gate(input_gate)
        output = self.main[0](img)
        output_gate_mult = self.gate_mult(output_gate)
        if self.opt.gate_affine:
            output_gate_add = self.gate_add(output_gate)
        for i in xrange(1,1+self.opt.ndres):
            alpha = output_gate_mult[:,(i-1)*self.opt.ndf:i*self.opt.ndf]
            alpha = alpha.resize(batchSize,self.opt.ndf,1,1)
            if self.opt.gate_affine:
                beta=output_gate_add[:,(i-1)*self.opt.ndf:i*self.opt.ndf]
                beta=beta.resize(batchSize,self.opt.ndf,1,1)
                output=self.main[i](output,alpha,beta)
            else:
                output=self.main[i](output,alpha)

        output = self.main_latter(output)
        return output


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




### Gated ResBlock

![](https://i.imgur.com/1NScbVw.png)

```python
class GatedResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super(GatedResnetBlock,self).__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        norm_layer='instance'
        # Submodules
        self.conv_0 = spectral_norm(nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias))
        if self.learned_shortcut:
            self.conv_s = spectral_norm( nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False))


    def forward(self, x,alpha=1.0,beta=0.0):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        #dx = self.norm(dx)
        if type(alpha)!=float:
            alpha=alpha.expand_as(x_s)
        if type(beta)!=float:
            beta=beta.expand_as(x_s)
        out = x_s + alpha*dx + beta   #x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s



class GatedConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=False):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1,dilation=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1,dilation=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8,res_op='add'):
    super(GatedConvResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride,use_sn=use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes,affine=True)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride , use_sn=use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups) ] #[nn.BatchNorm2d(planes,affine=True)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.res_op = res_op
  def forward(self, x,alpha=1.0,beta=0.0):
    residual = x
    if type(alpha)!=float:
        alpha=alpha.expand_as(x)
    if type(beta)!=float:
        beta= beta.expand_as(x)
    out = alpha*self.model(x) + beta
    out= residual_op(out,residual,self.res_op) #out += residual
    return out

class UpGatedConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8,res_op='add'):
    super(UpGatedConvResBlock, self).__init__()
    model = []
    model += upsampleLayer(inplanes , planes , upsample='nearest' , use_sn=use_sn)
    if norm_layer != 'none':
        model += [get_norm(planes,norm_layer,num_groups)]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    if norm_layer != 'none':
        model += [get_norm(planes,norm_layer,num_groups)] #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block +=  upsampleLayer(inplanes , planes , upsample='bilinear' , use_sn=use_sn)
    self.residual_block=nn.Sequential(*residual_block)
    self.res_op = res_op
  def forward(self, x, alpha=1.0,beta=0.0):
    residual = self.residual_block(x)
    f_x=self.model(x)
    if type(alpha)!=float:
        alpha=alpha.expand_as(f_x)
    if type(beta)!=float:
        beta=beta.expand_as(f_x)
    out = alpha * f_x + beta
    out = residual_op(out,residual,self.res_op)  #out += residual
    return out

class DownGatedConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False,norm_layer='batch',num_groups=8,res_op='add'):
    super(DownGatedConvResBlock, self).__init__()
    model = []
    model += downsampleLayer(inplanes,planes,downsample='avgpool',use_sn=use_sn)
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    if norm_layer != 'none':
        model += [ get_norm(planes,norm_layer,num_groups)  ]  #[nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block += downsampleLayer(inplanes,planes,downsample='avgpool',use_sn=use_sn)
    self.residual_block=nn.Sequential(*residual_block)
    self.res_op = res_op

  def forward(self, x,alpha=1.0,beta=0.0):
    residual = self.residual_block(x)
    f_x = self.model(x)
    if type(alpha)!=float:
        alpha=alpha.expand_as(f_x)
    if type(beta)!=float:
        beta=beta.expand_as(f_x)
    out = alpha * f_x + beta
    out = residual_op(out,residual,self.res_op) #out += residual
    return out

```



## Related

- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)

