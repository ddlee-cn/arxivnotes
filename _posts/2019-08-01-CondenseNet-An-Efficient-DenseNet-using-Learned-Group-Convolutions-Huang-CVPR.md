---
title: "CondenseNet: An Efficient DenseNet using Learned Group Convolutions - Huang - CVPR 2018"
tag:
- Image Classification
- CNN
redirect_from: /CondenseNet-An-Efficient-DenseNet-using-Learned-Group-Convolutions-Huang-CVPR.html
---



## Info
- Title: **CondenseNet: An Efficient DenseNet using Learned Group Convolutions**
- Task: **Image Classification**
- Author: Gao Huang, Shichen Liu, Laurens van der Maaten, Kilian Q. Weinberger
- Date: Nov. 2017
- Arxiv: [1711.09224](https://arxiv.org/abs/1711.09224)
- Published: CVPR 2018

## Highlights & Drawbacks
- Learned manner for group hyper-params
- Implementation with standard grouped convolutions


## Motivation & Design
**Group convolution**
![CleanShot 2019-08-18 at 11.25.54@2x](media/15598791313598/CleanShot%202019-08-18%20at%2011.25.54@2x.jpg)
Standard convolution (left) and group convolution (right). The latter enforces a sparsity pattern by partitioning the inputs (and outputs) into disjoint groups

![CleanShot 2019-08-18 at 11.27.49@2x](media/15598791313598/CleanShot%202019-08-18%20at%2011.27.49@2x.jpg)
Illustration of learned group convolutions with G = 3 groups and a condensation factor of C = 3. During training a fraction of (C − 1)/C connections are removed after each of the C − 1 condensing stages. Filters from the same group use the same set of features, and during test-time the index layer rearranges the features to allow the resulting model to be implemented as standard group convolutions.

## Performance & Ablation Study
Ablation study on CIFAR-10 to investigate the efficiency gains obtained by the various components of CondenseNet.
![CleanShot 2019-08-18 at 11.28.28@2x](media/15598791313598/CleanShot%202019-08-18%20at%2011.28.28@2x.jpg)

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


![CleanShot 2019-08-18 at 11.29.01@2x](media/15598791313598/CleanShot%202019-08-18%20at%2011.29.01@2x.jpg)
Actual inference time of different models on an ARM processor. All models are trained on ImageNet, and accept input with resolution 224×224.


## Code
[PyTorch](https://github.com/ShichenLiu/CondenseNet)



### Network

```python

class CondenseNet(nn.Module):
    def __init__(self, args):

        super(CondenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, args.num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
    
class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x
```



### DenseBlock & DenseLayer

```python
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)

```



```python
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        self.group_1x1 = args.group_1x1
        self.group_3x3 = args.group_3x3
        ### 1x1 conv i --> b*k
        self.conv_1 = LearnedGroupConv(in_channels, args.bottleneck * growth_rate,
                                       kernel_size=1, groups=self.group_1x1,
                                       condense_factor=args.condense_factor,
                                       dropout_rate=args.dropout_rate)
        ### 3x3 conv b*k --> k
        self.conv_2 = Conv(args.bottleneck * growth_rate, growth_rate,
                           kernel_size=3, padding=1, groups=self.group_3x3)

    def forward(self, x):
        x_ = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        return torch.cat([x_, x], 1)
```



## Related

- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/2019/08/17/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)
- [Convolutional Neural Network Must Reads: Xception, ShuffleNet, ResNeXt and DenseNet](https://arxivnote.ddlee.cn/2019/08/14/convolutional-neural-network-xception-shufflenet-resnext-densenet.html)