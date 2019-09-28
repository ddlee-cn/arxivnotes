---
title: "On the Effects and Weight Normalization in GAN - Xiang et al. - 2017 - PyTorch Code"
tag:
- GAN
- Normalization
---

## Info
- Title: **On the Effects and Weight Normalization in GAN**
- Author: Sitao Xiang, Hao Li
- Date: April. 2017
- Arxiv: [1704.03971](https://arxiv.org/abs/1704.0397)

## Highlights & Drawbacks
This paper explores the application of Weight Normalization in GAN. BN calculates the mean and variance at the mini-batch level, which is easy to introduce noise. It is not suitable for the generation model of GAN, and WN rewrites the parameters to introduce less noise.

## Motivation & Design
### 1. Propose T-ReLU with Affine Transformation to maintain the expressive power of the network after the introduction of WN
The naive parameterization layer has the following form:

$$
y=\frac{{w}^{T}x}{\|w\|}
$$

The layer in this form is referred to as "strict weight-normalized layer". If you change the linear layer to such a layer, the expressive power of the network will decrease, so you need to add the following affine transformation:

$$
y=\frac{{w}^{T}x}{\|w\|} \gamma + \beta
$$

Used to restore the expressive power of the network.

Bring the above transformation into ReLU, and after simplification, you can get the following T-ReLu:

$$
TReLU_\alpha (x) = ReLU(x-\alpha) + \alpha
$$

An important conclusion of the article is that after adding the affine transformation layer to the last layer of the network, the stacking "linear layer + ReLU" has the same expression ability as "strict weight-normalized layer + T-ReLU" (provided in the appendix) .

L below denotes a linear layer, R denotes ReLU, TR denotes TReLU, A denotes affine transformation, and S denotes the above-described strict weight-normalized layer.

The general idea is to add an affine transformation layer between ReLU and the linear layer. Due to the existence of the linear layer, the effect of affine transformation will be absorbed (equivalent to multiple linear layers or linear layers). constant. The structure of "L+R+A" can be equivalent to "S+TR+A". So recursively, you can get a conclusion. Personally think that it is equivalent to passing the bias in the linear layer into the threshold in TReLU (ie $\alpha$).

### 2. Presenting evaluation indicators for generated graphics

The generation effect of a generative model is often difficult to evaluate. The result given by DcGAN is also a comparison of the generated images. In this paper, an index for evaluating the production effect is proposed and is consistent with the subjective evaluation of people.

The specific indicator of the evaluation is the Euclidean distance between the generated image and the test set image. The object of the evaluation is that the generator is Generator. Has the following form:

$$
\frac{1}{m} \sum_{i=1}^{m} min_z {\|G(z)-x^{(i)}\|}^2
$$

Among them, $min$ means to use the gradient descent method to make the best image generation. But in fact, this is expensive.

## Code
[PyTorch](https://github.com/stormraiser/GAN-weight-norm)

**T-ReLU**

```python
class TPReLU(Module):

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(TPReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.bias = Parameter(torch.zeros(num_parameters))

    def forward(self, input):
    	bias_resize = self.bias.view(1, self.num_parameters, *((1,) * (input.dim() - 2))).expand_as(input)
        return F.prelu(input - bias_resize, self.weight.clamp(0, 1)) + bias_resize


```

*Weigh-normalized layer*

`````python
class WeightNormalizedLinear(Module):

    def __init__(self, in_features, out_features, scale=True, bias=True, init_factor=1, init_scale=1):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)
        if scale:
            self.scale = Parameter(torch.Tensor(1, out_features).fill_(init_scale))
        else:
            self.register_parameter('scale', None)
        self.reset_parameters(init_factor)

    def reset_parameters(self, factor):
        stdv = 1. * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def weight_norm(self):
        return self.weight.pow(2).sum(1).add(1e-6).sqrt()

    def norm_scale_bias(self, input):
        output = input.div(self.weight_norm().transpose(0, 1).expand_as(input))
        if self.scale is not None:
            output = output.mul(self.scale.expand_as(input))
        if self.bias is not None:
            output = output.add(self.bias.expand_as(input))
        return output

    def forward(self, input):
        return self.norm_scale_bias(F.linear(input, self.weight))
```

Observing the forward function above, it can be found that TReLU adds the learned parameter of bias, while the weight-normalized layer standardizes the incoming weight.

