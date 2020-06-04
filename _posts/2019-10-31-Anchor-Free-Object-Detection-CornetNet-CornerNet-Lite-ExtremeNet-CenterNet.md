---
title: "Anchor-Free Object Detection(Part 1): CornerNet, CornerNet-Lite, ExtremeNet, CenterNet"
tag:
- Object Detection
---

## CornerNet: Detecting Objects as Paired Keypoints - ECCV 2018

![CornerNet: Detecting Objects as Paired Keypoints - ECCV 2018](https://i.imgur.com/npESC4v.jpg)

The model detect an object as a pair of bounding box corners grouped together. A convolutional network outputs a heatmap for all top-left corners, a heatmap for all bottom-right corners, and an embedding vector for each detected corner. The network is trained to predict similar embeddings for corners that belong to the same object.

![CornerNet: Detecting Objects as Paired Keypoints - ECCV 2018](https://i.imgur.com/wHrOzEW.jpg)

The backbone network is followed by two prediction modules, one for the top-left corners and the other for the bottom-right corners. Using the predictions from both modules, we locate and group the corners.

![CornerNet: Detecting Objects as Paired Keypoints - ECCV 2018](https://i.imgur.com/YyHEujb.jpg)


### Code

[PyTorch](https://github.com/princeton-vl/CornerNet)

```python
class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class tl_pool(pool):
    def __init__(self, dim):
        super(tl_pool, self).__init__(dim, TopPool, LeftPool)

class br_pool(pool):
    def __init__(self, dim):
        super(br_pool, self).__init__(dim, BottomPool, RightPool)

def make_tl_layer(dim):
    return tl_pool(dim)

def make_br_layer(dim):
    return br_pool(dim)

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class model(kp):
    def __init__(self, db):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = 80

        super(model, self).__init__(
            n, 2, dims, modules, out_dim,
            make_tl_layer=make_tl_layer,
            make_br_layer=make_br_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

loss = AELoss(pull_weight=1e-1, push_weight=1e-1, focal_loss=_neg_loss)
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




## CornerNet-Lite: Efficient Keypoint Based Object Detection


CornerNet-Saccade speeds up inference by reducing the number of pixels to process. It uses an attention mechanism similar to saccades in human vision. It starts with a downsized full image and generates an attention map, which is then zoomed in on and processed further by the model. This differs from the original CornerNet in that it is applied fully convolutionally across multiple scales.

![CornerNet-Lite: Efficient Keypoint Based Object Detection](https://i.imgur.com/Zzu1wRl.jpg)

We predict a set of possible object locations from the attention maps and bounding boxes generated on a downsized full image. We zoom into each location and crop a small region around that location. Then we detect objects in each region. We control the efficiency by ranking the object locations and choosing top k locations to process. Finally, we merge the detections by NMS.

![CornerNet-Lite: Efficient Keypoint Based Object Detection](https://i.imgur.com/DTkcDKz.png)

CornerNet-Squeeze speeds up inference by reducing the amount of processing per pixel. It incorporates ideas from SqueezeNet and MobileNets, and introduces a new, compact hourglass backbone that makes extensive use of 1×1 convolution, bottleneck layer, and depth-wise separable convolution.

![CornerNet-Lite: Efficient Keypoint Based Object Detection](https://i.imgur.com/iXqDI10.png)

### Code

[PyTorch](https://github.com/princeton-vl/CornerNet-Lite)

#### CornerNet-Saccade

```python
def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

class model(saccade_net):
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

    def __init__(self):
        stacks  = 3
        pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(128, 256, stride=2)
        )
        hg_mods = nn.ModuleList([
            saccade_module(
                3, [256, 384, 384, 512], [1, 1, 1, 1],
                make_pool_layer=make_pool_layer,
                make_hg_layer=make_hg_layer
            ) for _ in range(stacks)
        ])
        cnvs    = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters  = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])

        att_mods = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    convolution(3, 384, 256, with_bn=False),
                    nn.Conv2d(256, 1, (1, 1))
                ),
                nn.Sequential(
                    convolution(3, 384, 256, with_bn=False),
                    nn.Conv2d(256, 1, (1, 1))
                ),
                nn.Sequential(
                    convolution(3, 256, 256, with_bn=False),
                    nn.Conv2d(256, 1, (1, 1))
                )
            ]) for _ in range(stacks)
        ])
        for att_mod in att_mods:
            for att in att_mod:
                torch.nn.init.constant_(att[-1].bias, -2.19)

        hgs = saccade(pre, hg_mods, cnvs, inters, cnvs_, inters_) 

        tl_modules = nn.ModuleList([corner_pool(256, TopPool, LeftPool) for _ in range(stacks)])
        br_modules = nn.ModuleList([corner_pool(256, BottomPool, RightPool) for _ in range(stacks)])

        tl_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        br_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        for tl_heat, br_heat in zip(tl_heats, br_heats):
            torch.nn.init.constant_(tl_heat[-1].bias, -2.19)
            torch.nn.init.constant_(br_heat[-1].bias, -2.19)

        tl_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        br_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])

        tl_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
        br_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])

        super(model, self).__init__(
            hgs, tl_modules, br_modules, tl_heats, br_heats, 
            tl_tags, br_tags, tl_offs, br_offs, att_mods
        )

        self.loss = CornerNet_Saccade_Loss(pull_weight=1e-1, push_weight=1e-1)
```

#### CornerNet-Squeeze

```python
class fire_module(nn.Module):
    def __init__(self, inp_dim, out_dim, sr=2, stride=1):
        super(fire_module, self).__init__()
        self.conv1    = nn.Conv2d(inp_dim, out_dim // sr, kernel_size=1, stride=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_dim // sr)
        self.conv_1x1 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=1, stride=stride, bias=False)
        self.conv_3x3 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=3, padding=1, 
                                  stride=stride, groups=out_dim // sr, bias=False)
        self.bn2      = nn.BatchNorm2d(out_dim)
        self.skip     = (stride == 1 and inp_dim == out_dim)
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        conv2 = torch.cat((self.conv_1x1(bn1), self.conv_3x3(bn1)), 1)
        bn2   = self.bn2(conv2)
        if self.skip:
            return self.relu(bn2 + x)
        else:
            return self.relu(bn2)

def make_pool_layer(dim):
    return nn.Sequential()

def make_unpool_layer(dim):
    return nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

def make_layer(inp_dim, out_dim, modules):
    layers  = [fire_module(inp_dim, out_dim)]
    layers += [fire_module(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

def make_layer_revr(inp_dim, out_dim, modules):
    layers  = [fire_module(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [fire_module(inp_dim, out_dim)]
    return nn.Sequential(*layers)

def make_hg_layer(inp_dim, out_dim, modules):
    layers  = [fire_module(inp_dim, out_dim, stride=2)]
    layers += [fire_module(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

class model(hg_net):
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(1, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

    def __init__(self):
        stacks  = 2
        pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(128, 256, stride=2),
            residual(256, 256, stride=2)
        )
        hg_mods = nn.ModuleList([
            hg_module(
                4, [256, 256, 384, 384, 512], [2, 2, 2, 2, 4],
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_up_layer=make_layer,
                make_low_layer=make_layer,
                make_hg_layer_revr=make_layer_revr,
                make_hg_layer=make_hg_layer
            ) for _ in range(stacks)
        ])
        cnvs    = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters  = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])

        hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_) 

        tl_modules = nn.ModuleList([corner_pool(256, TopPool, LeftPool) for _ in range(stacks)])
        br_modules = nn.ModuleList([corner_pool(256, BottomPool, RightPool) for _ in range(stacks)])

        tl_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        br_heats = nn.ModuleList([self._pred_mod(80) for _ in range(stacks)])
        for tl_heat, br_heat in zip(tl_heats, br_heats):
            torch.nn.init.constant_(tl_heat[-1].bias, -2.19)
            torch.nn.init.constant_(br_heat[-1].bias, -2.19)

        tl_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])
        br_tags = nn.ModuleList([self._pred_mod(1) for _ in range(stacks)])

        tl_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])
        br_offs = nn.ModuleList([self._pred_mod(2) for _ in range(stacks)])

        super(model, self).__init__(
            hgs, tl_modules, br_modules, tl_heats, br_heats, 
            tl_tags, br_tags, tl_offs, br_offs
        )

        self.loss = CornerNet_Loss(pull_weight=1e-1, push_weight=1e-1)
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



## (ExtremeNet)Bottom-up Object Detection by Grouping Extreme and Center Points - CVPR 2019

![(ExtremeNet)Bottom-up Object Detection by Grouping Extreme and Center Points - CVPR 2019](https://i.imgur.com/AsbOUJw.png)

In this paper, we propose ExtremeNet, a bottom-up object detection framework that detects four extreme points (top-most, left-most, bottom-most, right-most) of an object. We use a state-of-the-art keypoint estimation framework to find extreme points, by predicting four multi-peak heatmaps for each object category. In addition, we use one heatmap per category predicting the object center, as the average of two bounding box edges in both the x and y dimension. We group extreme points into objects with a purely geometry-based approach. We group four extreme points, one from each map, if and only if their geometric center is predicted in the center heatmap with a score higher than a pre-defined threshold. We enumerate all $O(n^4)$combinations of extreme point prediction, and select the valid ones.

![(ExtremeNet)Bottom-up Object Detection by Grouping Extreme and Center Points - CVPR 2019](https://i.imgur.com/Ek7WhEw.png)

Given four extreme points t, b, r, l extracted from heatmaps Ŷ (t) , Ŷ (l) , Ŷ (b) , Ŷ (r), we compute their geometric center $c=\left(\frac{l_{x}+t_{x}}{2}, \frac{t_{y}+b_{y}}{2}\right)$. If this center is predicted2 2with a high response in the center map Ŷ (c), we commit the extreme points as a valid detection: Ŷcx ,cy ≥ τc for a threshold τc. We then enumerate over all quadruples of keypoints t, b, r, l in a brute force manner. We extract detections foreach class independently. 


![(ExtremeNet)Bottom-up Object Detection by Grouping Extreme and Center Points - CVPR 2019](https://i.imgur.com/TYhe4Al.png)



### Code

[PyTorch](https://github.com/xingyizhou/ExtremeNet)

```python
class exkp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=None, make_br_layer=None,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(exkp, self).__init__()

        self.nstack    = nstack
        self._decode   = _exct_decode

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.t_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.l_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.b_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.r_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.ct_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        for t_heat, l_heat, b_heat, r_heat, ct_heat in \
          zip(self.t_heats, self.l_heats, self.b_heats, \
              self.r_heats, self.ct_heats):
            t_heat[-1].bias.data.fill_(-2.19)
            l_heat[-1].bias.data.fill_(-2.19)
            b_heat[-1].bias.data.fill_(-2.19)
            r_heat[-1].bias.data.fill_(-2.19)
            ct_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.t_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.l_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.b_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.r_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)

    def _train(self, *xs):
        image  = xs[0]
        t_inds = xs[1]
        l_inds = xs[2]
        b_inds = xs[3]
        r_inds = xs[4]

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,
            self.t_heats, self.l_heats, self.b_heats, self.r_heats,
            self.ct_heats,
            self.t_regrs, self.l_regrs, self.b_regrs, self.r_regrs,
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            t_heat_, l_heat_, b_heat_, r_heat_ = layer[2:6]
            ct_heat_                           = layer[6]
            t_regr_, l_regr_, b_regr_, r_regr_ = layer[7:11]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            t_heat, l_heat = t_heat_(cnv), l_heat_(cnv)
            b_heat, r_heat = b_heat_(cnv), r_heat_(cnv)
            ct_heat        = ct_heat_(cnv)

            t_regr, l_regr = t_regr_(cnv), l_regr_(cnv)
            b_regr, r_regr = b_regr_(cnv), r_regr_(cnv)

            t_regr = _tranpose_and_gather_feat(t_regr, t_inds)
            l_regr = _tranpose_and_gather_feat(l_regr, l_inds)
            b_regr = _tranpose_and_gather_feat(b_regr, b_inds)
            r_regr = _tranpose_and_gather_feat(r_regr, r_inds)

            outs += [t_heat, l_heat, b_heat, r_heat, ct_heat, \
                     t_regr, l_regr, b_regr, r_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs
```

```python

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class model(exkp):
    def __init__(self, db):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = 80

        super(model, self).__init__(
            n, 2, dims, modules, out_dim,
            make_tl_layer=None,
            make_br_layer=None,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

loss = CTLoss(focal_loss=_neg_loss)
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



## (CenterNet-D)CenterNet: Keypoint Triplets for Object Detection - ICCV 2019

![(CenterNet-D)CenterNet: Keypoint Triplets for Object Detection - ICCV 2019](https://i.imgur.com/6pFsCoQ.png)

In this paper, we present a low-cost yet effective solution named CenterNet, which explores the central part of a proposal, i.e., the region that is close to the geometric center of a box, with one extra keypoint. We intuit that if a predicted bounding box has a high IoU with the ground-truth box, then the probability that the center keypoint in the central region of the bounding box will be predicted as the same class is high, and vice versa. Thus, during inference, after a proposal is generated as a pair of corner keypoints, we determine if the proposal is indeed an object by checking if there is a center keypoint of the same class falling within its central region.

![(CenterNet-D)CenterNet: Keypoint Triplets for Object Detection - ICCV 2019](https://i.imgur.com/MFkqXVY.jpg)

A convolutional backboneoutput two corner heatmaps and a center keypoint heatmap, respectively.the similar embeddings are used to detect a potential boundingthe final bounding boxes. network applies cascade corner pooling and center pooling to Similar to CornerNet, a pair of detected corners and box. Then the detected center keypoints are used to determine the final bounding boxes.

![(CenterNet-D)CenterNet: Keypoint Triplets for Object Detection - ICCV 2019](https://i.imgur.com/ozuZHNW.jpg)



### Code

[PyTorch](https://github.com/Duankaiwen/CenterNet)

```python

class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

        self.look_conv1 = convolution(3, dim, 128)
        self.look_conv2 = convolution(3, dim, 128)
        self.P1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.P2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        # pool 1
        look_conv1   = self.look_conv1(x)
        p1_conv1     = self.p1_conv1(x)
        look_right   = self.pool2(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right)
        pool1        = self.pool1(P1_look_conv)

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down   = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2    = self.pool2(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class pool_cross(nn.Module):
    def __init__(self, dim, pool1, pool2, pool3, pool4):
        super(pool_cross, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.pool3 = pool3()
        self.pool4 = pool4()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)
        pool1    = self.pool3(pool1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)
        pool2    = self.pool4(pool2)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class tl_pool(pool):
    def __init__(self, dim):
        super(tl_pool, self).__init__(dim, TopPool, LeftPool)

class br_pool(pool):
    def __init__(self, dim):
        super(br_pool, self).__init__(dim, BottomPool, RightPool)

class center_pool(pool_cross):
    def __init__(self, dim):
        super(center_pool, self).__init__(dim, TopPool, LeftPool, BottomPool, RightPool)

def make_tl_layer(dim):
    return tl_pool(dim)

def make_br_layer(dim):
    return br_pool(dim)

def make_ct_layer(dim):
    return center_pool(dim)

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class model(kp):
    def __init__(self, db):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = 80

        super(model, self).__init__(
            db, n, 1, dims, modules, out_dim,
            make_tl_layer=make_tl_layer,
            make_br_layer=make_br_layer,
            make_ct_layer=make_ct_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

loss = AELoss(pull_weight=1e-1, push_weight=1e-1, focal_loss=_neg_loss)
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



## (CenterNet-Z)Objects as Points

![(CenterNet-Z)Objects as Points](https://i.imgur.com/OCXfp70.png)

We represent objects by a single point at their bounding box center. Other properties, such as object size, dimension, 3D extent, orientation, and pose are then regressed directly from image features at the center location. Object detection is then a standard keypoint estimation problem. We simply feed the input image to a fully convolutional network that generates a heatmap. Peaks in this heatmap correspond to object centers. Image features at each peak predict the objects bounding box height and weight. The model trains using standard dense supervised learning. Inference is a single network forward-pass, without non-maximal suppression for post-processing.

![(CenterNet-Z)Objects as Points](https://i.imgur.com/qwbBPGi.png)

The numbers in the boxes represent the stride to the image. (a): Hourglass Network, We
use it as is in CornerNet.. (b): ResNet with transpose convolutions. We add one 3 × 3 deformable convolutional layer before each up-sampling layer. Specifically, we first use deformable convolution to change the channels and then use transposed convolution to upsample the feature map (such two steps are shown separately in 32 → 16. We show these two steps together as a dashed arrow for 16 → 8 and 8 → 4). (c): The original DLA-34 for semantic segmentation. (d): Our modified DLA-34. We add more skip connections from the bottom layers and upgrade every convolutional layer in upsampling stages to deformable convolutional layer.

!(CenterNet-Z)Objects as Points[](https://i.imgur.com/GA6wNG7.png)


### Code

[PyTorch](https://github.com/xingyizhou/CenterNet)


```python
def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections
```

## Related

- [Object Detection Must Reads(1): Fast RCNN, Faster RCNN, R-FCN and FPN](https://arxivnote.ddlee.cn/object-detectin-fast-rcnn-faster-rcnn-rfcn.html)

- [Object Detection Must Reads(2): YOLO, YOLO9000, and RetinaNet](https://arxivnote.ddlee.cn/object-detection-must-reads-yolo-retinanet.html)

- [Object Detection Must Reads(3): SNIP, SNIPER, OHEM, and DSOD](https://arxivnote.ddlee.cn/2019/08/13/Object-Detection-SNIPER-DSOD-OHEM.html)

- [RoIPooling in Object Detection: PyTorch Implementation(with CUDA)](https://cvnote.ddlee.cn/2019/09/16/RoIPooling-PyTorch.html)

- [Bounding Box(BBOX) IOU Calculation and Transformation in PyTorch](https://cvnote.ddlee.cn/2019/09/09/Bounding-Box-Transform.html)

- [Object Detection: Anchor Generator in PyTorch](https://cvnote.ddlee.cn/2019/09/07/PyTorch-Anchor-Generator.html)

- [Assign Ground Truth to Anchors in Object Detection with Python](https://cvnote.ddlee.cn/2019/09/21/Assign-Ground-Truth-Object-Detection.html)

- [From Classification to Panoptic Segmentation: 7 years of Visual Understanding with Deep Learning](https://arxivnote.ddlee.cn/Classification-to-Panoptic-Segmentation-visual-understanding-CVPR.html)

- [Convolutional Neural Network Must Reads: Xception, ShuffleNet, ResNeXt and DenseNet](https://arxivnote.ddlee.cn/convolutional-neural-network-xception-shufflenet-resnext-densenet.html)

- [Anchor-Free Object Detection(Part 1): CornerNet, CornerNet-Lite, ExtremeNet, CenterNet](https://arxivnote.ddlee.cn/2019/10/31/Anchor-Free-Object-Detection-CornetNet-CornerNet-Lite-ExtremeNet-CenterNet.html)

- [Anchor-Free Object Detection(Part 2): FSAF, FoveaBox, FCOS, RepPoints](https://arxivnote.ddlee.cn/2019/10/31/Anchor-Free-Object-Detection-FSAF-FoveaBox-FCOS-RepPoints.html)

  