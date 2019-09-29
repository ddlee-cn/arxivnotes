---
title: "Convolutional Neural Network Must Reads: Xception, ShuffleNet, ResNeXt and DenseNet"
tag:
- Image Classification
- CNN
redirect_from: /convolutional-neural-network-xception-shufflenet-resnext-densenet.html
---







## Xception: Deep Learning with Depthwise Seperable Convolutions - Chollet et al. - 2016
### Info
- Title: **Xception: Deep Learning with Depthwise Seperable Convolutions**
- Author: F. Chollet
- Arxiv: [1610.02357](https://arxiv.org/abs/1610.02357)
- Date: Oct. 2016

### Highlights & Drawbacks
Replaced 1×1 convolution and 3×3 convolution in Inception unit with Depth-wise seperable convolution

### Motivation & Design

The article points out that the assumption behind the Inception unit is that the correlation between the channel and the space can be fully decoupled, similarly the convolution structure in the length and height directions (the 3 × 3 convolution in Inception-v3 is 1 × 3 and 3 × 1 convolution replacement).

Further, Xception is based on a stronger assumption: the correlation between channels and cross-space is completely decoupled. This is also the concept modeled by Depthwise Separable Convolution.
A simple Inception Module:

![Xception: Deep Learning with Depthwise Seperable Convolutions](https://i.imgur.com/voGGEeh.png)


is equal to:

![Xception: Deep Learning with Depthwise Seperable Convolutions](https://i.imgur.com/ttldnjQ.png)



Push # of channel to extreme, we obtain Depthwise Separable Convolution:

![Xception: Deep Learning with Depthwise Seperable Convolutions](https://i.imgur.com/2AuC4j9.png)


[NetScope Visualization](http://ethereon.github.io/netscope/#gist/931d7c91b22109f83bbbb7ff1a215f5f) and source code: [awesome_cnn](https://github.com/ddlee96/awesome_cnn).

![Xception: Deep Learning with Depthwise Seperable Convolutions](https://i.imgur.com/BT6sHIb.png)

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

## ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design - Ma - ECCV 2018

### Info
- Title: **ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design**
- Task: **Image Classification**
- Author: N. Ma, X. Zhang, H.-T. Zheng, and J. Sun
- Date: Jul. 2018
- Arxiv: [1807.11164](https://arxiv.org/abs/1807.11164)
- Published: ECCV 2018

### Highlights & Drawbacks
- Detailed analysis from hardware perspective
- Design guidelines for efficient CNN architecture


### Motivation & Design
![ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://i.imgur.com/KDc3spp.png)

There is discrepancy between the indirect (FLOPs) and direct (speed) metrics. FLOPs doesn't take factors like memory access cost (MAC) and degree of parallelism into account, which is critical in actual calculation process.

The authors proposed four guidelines for efficient CNN architecture design, based on analysis from hardware and processing framework's perspective:
1. Equal channel width minimizes memory access cost (MAC)
![ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://i.imgur.com/NH1tzX9.png)
(c1 and c2 closer, faster)

2. Excessive group convolution increases MAC
![ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://i.imgur.com/xIg13Di.png)
(larger group get lower speed)

3. Network fragmentation reduces degree of parallelism
![ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://i.imgur.com/wbWu3Ku.png)

4. Element-wise operations are non-negligible
![ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://i.imgur.com/ILD1CIa.png)

**The final ShuffleNet v2 Unit**
![ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://i.imgur.com/RfsZ6Qk.png)


## Performance & Ablation Study

**Image classification results**
![ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://i.imgur.com/HS0sag7.png)

**Object detection results**
![ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://i.imgur.com/X98rMbF.png)


## Code

[PyTorch](https://github.com/ericsun99/Shufflenet-v2-Pytorch)

**Channel Shuffle**

```python
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
```

**Inverted Residual Module**

```python
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)
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

## (ResNeXt)Aggregated Residual Transformations for Deep Neural Networks - Xie et al. - CVPR 2017

### Info
- Title: **Aggregated Residual Transformations for Deep Neural Networks**
- Task: **Image Classification**
- Author: S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. H
- Arxiv: [1611.05431](https://arxiv.org/abs/1611.05431)
- Date: Nov. 2016
- Published: CVPR 2017
- 1st Runner Up in ILSVRC 2016

### Highlights & Drawbacks
The core idea of ResNeXt is normalizing the multi-path structure of Inception module. Instead of using hand-designed 1x1, 3x3, and 5x5 convolutions, ResNeXt proposed a new hyper-parameter with reasonable meaning for network design.

The authors proposed a new dimension on designing neural network, which is called *cardinality*. Besides # of layers, # of channels, cardinality describes the count of paths inside one module. Compared to the Inception model, the paths share the exactly the same hyper-parameter. Additionally, short connection is added between layers.


### Motivation & Design

The three classical pattern on designing a neural network:

- **Repeat**: Starting with AlexNet and VGG, repeating the same structure is one of the most popular patterns of deep networks.
- **Multi-path**: Presented by the Inception-Series. Splitting inputs, transforming with multiple-size convolutions, then concatenation.
- **Skip-connection**: Applied to Image Recognition by ResNet. Simply rewriting the target function into identity mapping and residual function, allowing the interaction between shallow layers and deep layers.

The residual function is rewritten into:
$$
\mathbf{y}=\mathbf{x}+\sum_{i=1}^{C} \mathcal{T}_{i}(\mathbf{x}),
$$
C denotes the number of transformations(paths) inside the layer, a.k.a. cardinality.

![(ResNeXt)Aggregated Residual Transformations for Deep Neural Networks](https://i.imgur.com/JxJJiOH.png)


As the number of paths increases, the number of channel for each path is reduced to maintain capacity of network.

[NetScope Visualization](http://ethereon.github.io/netscope/#/gist/c2ba521fcb60520abb0b0da0e9c0f2ef) and source code(Pytorch+Caffe):[awesome_cnn](https://github.com/ddlee96/awesome_cnn).

### Performance & Ablation Study

![(ResNeXt)Aggregated Residual Transformations for Deep Neural Networks](https://i.imgur.com/I4Nhs1X.png)

![(ResNeXt)Aggregated Residual Transformations for Deep Neural Networks](https://i.imgur.com/FYvIp3v.png)


### Code
[Torch](https://github.com/facebookresearch/ResNeXt)

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

## (DenseNet)Densely Connected Convolutional Networks - Huang - CVPR 2017
### Info
- Title: **Densely Connected Convolutional Network**
- Task: **Image Classification**
- Author: Gao Huang, Zhuang Liu, Laurens van der Maaten and Kilian Weinberger 
- Arxiv: [1608.06993](https://arxiv.org/abs/1608.06993)
- Published: CVPR 2017(Best Paper Award)

### Highlights

DenseNet takes the idea of ​​shortcut-connection to its fullest. Inside a DenseBlock, the output of each layer is created with the following layers. It is important to note that unlike the addition in ResNet, the DenseNet connection shortcut is Concat, so the deeper the layer, the more the input channel number. Big.


### Motivation & Design

![(DenseNet)Densely Connected Convolutional Networks](https://i.imgur.com/WTz22Su.png)

The entire network is divided into Dense Block and Transition Layer. The former is densely connected internally and maintains the same size feature map. The latter is the connection layer between DenseBlocks and performs the downsampling operation.

Within each DenseBlock, the accepted data dimension will become larger as the number of layers deepens (because the output of the previous layer is spliced ​​continuously), and the rate of growth is the initial channel number. The article calls the channel number as the growth rate. A hyper-parameter of the model. When the initial growth rate is 32, the number of channels in the last layer will increase to 1024 under the DenseNet121 architecture.

[Netscope Visualization] (http://ethereon.github.io/netscope/#/gist/56cb18697f42eb0374d933446f45b151) and source code: [awesome_cnn](https://github.com/ddlee96/awesome_cnn).

### Performance & Ablation Study

The authors have done experiments on both CIFAR and ImageNet. DenseNet has achieved comparable performance with ResNet. After adding Botleneck and a part of the compression technique, it can achieve the same effect as ResNet with fewer parameters:

![(DenseNet)Densely Connected Convolutional Networks](https://i.imgur.com/s58rsYr.png)

### Code
[Caffe](https://github.com/liuzhuang13/DenseNet)

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
- [Object Detection Must Reads(3): SNIP, SNIPER, OHEM, and DSOD](https://arxivnote.ddlee.cn/object-detection-must-reads-snip-sniper-ohem-dsod.html)
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)
- [Image to Image Translation(1): pix2pix, S+U, CycleGAN, UNIT, BicycleGAN, and StarGAN](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pix-CycleGAN-UNIT-BicycleGAN-StarGAN.html)
- [Image to Image Translation(2): pix2pixHD, MUNIT, DRIT, vid2vid, SPADE, INIT, and FUNIT](https://arxivnote.ddlee.cn/Image-to-image-Translation-pix2pixHD-MUNIT-DRIT-vid2vid-SPADE-INIT-FUNIT.html)