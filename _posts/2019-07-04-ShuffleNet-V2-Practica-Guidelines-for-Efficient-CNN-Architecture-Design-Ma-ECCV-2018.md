---
title: "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design - Ma - ECCV 2018"
tag:
- Object Detection
- Image Classification
---

## Info
- Title: **ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design**
- Task: **Image Classification**
- Author: N. Ma, X. Zhang, H.-T. Zheng, and J. Sun
- Date: Jul. 2018
- Arxiv: [1807.11164](https://arxiv.org/abs/1807.11164)
- Published: ECCV 2018

## Highlights & Drawbacks
- Detailed analysis from hardware perspective
- Design guidelines for efficient CNN architecture


<!-- more -->


## Motivation & Design
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