---
title: Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019
tag:
- Video
- Self-Supervised
- Representation Learning
---

## Info

- Title: Learning Correspondence from the Cycle-Consistency of Time
- Task: Representation Learning
- Author: Xiaolong Wang, Allan Jabri, Alexei A. Efros
- Date: Mar. 2019
- Arxiv: [1903.07593](https://arxiv.org/abs/1903.07593)
- Published: CVPR 2019



## Highlights

- Self-supervised for representation learning inside single video
- The learned representation can be generalized to video object segmentation, keypoint tracking, optical flow(high-level and low-level)



## Abstract

We introduce a self-supervised method for learning visual correspondence from unlabeled video. The main idea is to use cycle-consistency in time as free supervisory signal for learning visual representations from scratch. At training time, our model learns a feature map representation to be useful for performing cycle-consistent tracking. At test time, we use the acquired representation to find nearest neighbors across space and time. We demonstrate the generalizability of the representation -- without finetuning -- across a range of visual correspondence tasks, including video object segmentation, keypoint tracking, and optical flow. Our approach outperforms previous self-supervised methods and performs competitively with strongly supervised methods.



## Motivation & Design



### Motivation



![Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](<https://i.imgur.com/5fPPjic.png>)

Most fundamental vision problems, from optical flow and tracking to action recognition and 3D reconstruction, require some notion of visual correspondence. Correspondence is the glue that links disparate visual percepts into persistent entities and underlies visual reasoning in space and time.

We can obtain unlimited supervision for correspondence by tracking backward and then forward (i.e. along a cycle in time) and using the inconsistency between the start and end points as the loss function.



### The Challenge

![Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://i.imgur.com/TnWMsj9.png)



### Approach


The goal is to learn a feature space  $φ$  by tracking a patch pt extracted from image It backwards and then forwards in time, while minimizing the cycle-consistency loss  $l_{\theta}$  (yellow arrow). Learning  $φ$  relies on a simple tracking operation T , which takes as inputs the features of a current patch and a target image, and returns the image feature region with maximum similarity. Without information of where the patch came from, T must match features encoded by  $φ$  to localize the next patch. As shown in (a), T can be iteratively applied backwards and then forwards through time to track along an arbitrarily long cycle. The cycle-consistency loss  $l_{\theta}$  is the euclidean distance between the spatial coordinates of initial patch pt and the patch found at the end of the cycle in It. In order to minimize  $l_{\theta}$ , the model must learn a feature space  $φ$  that allows for robustly measuring visual similarity between patches along the cycle.

Note that T is only used in training and is deliberately designed to be weak, so as to place the burden of representation on  $φ$ . At test time, the learned  $φ$  is used directly for computing correspondences.




![Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://i.imgur.com/0KVbleH.png)



(a) During training, the model learns a feature space encoded by $φ$ to perform tracking using tracker $T$. By tracking backward and then forward, we can use cycle-consistency to supervise learning of $φ$. Note that only the initial patch $p_t$ is explicitly encoded by $φ$; other patch features along the cycle are obtained by localizing image features. 

(b) We show one step of tracking back in time from $t$ to $t − 1$. Given input image features $x^I_{t−1}$ and query patch features $x^p_t$ , T localizes the patch $x^p_{t−1}$ in $x^I_{t−1}$ . This operation is performed iteratively to track along the cycle in (a).



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




### Loss Functions

**Tracking**

$$
\mathcal{L}_{l o n g}^{i}=l_{\theta}\left(x_{t}^{p}, \mathcal{T}^{(i)}\left(x_{t-i+1}^{I}, \mathcal{T}^{(-i)}\left(x_{t-1}^{I}, x_{t}^{p}\right)\right)\right)
$$




**Skip Cycle**

$$
\mathcal{L}_{s k i p}^{i}=l_{\theta}\left(x_{t}^{p}, \mathcal{T}\left(x_{t}^{I}, \mathcal{T}\left(x_{t-i}^{I}, x_{t}^{p}\right)\right)\right)
$$




**Feature Similarity**

$$
\mathcal{L}_{s i m}^{i}=-\left\langle x_{t}^{p}, \mathcal{T}\left(x_{t-i}^{I}, x_{t}^{p}\right)\right\rangle
$$





## Experiments & Ablation Study



![Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://i.imgur.com/WfWvgFn.png)



### Instance Mask Evaluation

![Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://i.imgur.com/v2FzqNr.png)





### Keypoints Evaluation



![Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://i.imgur.com/vnaLuK8.png)

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


more on [YouTube Video](https://www.youtube.com/watch?v=yLUCTX7bZVI).




## Code

[Project Site](<https://ajabri.github.io/timecycle/>)

[PyTorch](https://github.com/xiaolonw/TimeCycle)

### Model Definition

```python
class CycleTime(nn.Module):

    def __init__(self, class_num=8, dim_in=2048, trans_param_num=3, detach_network=False, pretrained=True, temporal_out=4, T=None, hist=1):
        super(CycleTime, self).__init__()

        dim = 512
        print(pretrained)

        resnet = resnet_res4s1.resnet50(pretrained=pretrained)
        self.encoderVideo = inflated_resnet.InflatedResNet(copy.deepcopy(resnet))
        self.detach_network = detach_network
        self.hist = hist

        self.div_num = 512
        self.T = self.div_num**-.5 if T is None else T
        print('self.T:', self.T)

        self.afterconv1 = nn.Conv3d(1024, 512, kernel_size=1, bias=False)

        self.spatial_out1 = 30
        self.spatial_out2 = 10
        self.temporal_out = temporal_out

        self.afterconv3_trans = nn.Conv2d(self.spatial_out1 * self.spatial_out1, 128, kernel_size=4, padding=0, bias=False)
        self.afterconv4_trans = nn.Conv2d(128, 64, kernel_size=4, padding=0, bias=False)

        corrdim = 64 * 4 * 4
        corrdim_trans = 64 * 4 * 4

        self.linear2 = nn.Linear(corrdim_trans, trans_param_num)

        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.avgpool3d = nn.AvgPool3d((4, 2, 2), stride=(1, 2, 2))
        self.maxpool2d = nn.MaxPool2d(2, stride=2)


        # initialization

        nn.init.kaiming_normal_(self.afterconv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.afterconv3_trans.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.afterconv4_trans.weight, mode='fan_out', nonlinearity='relu')

        # assuming no fc pre-training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # transformation
        self.geometricTnf = GeometricTnfAffine(geometric_model='affine',
                                         tps_grid_size=3,
                                         tps_reg_factor=0.2,
                                         out_h=self.spatial_out2, out_w=self.spatial_out2,
                                         offset_factor=227/210)

        xs = np.linspace(-1,1,80)
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        self.xs = xs

        self.criterion_inlier = WeakInlierCountPool(geometric_model='affine', tps_grid_size=3, tps_reg_factor=0.2, h_matches=30, w_matches=30, use_conv_filter=False, dilation_filter=0, normalize_inlier_count=True)
        self.criterion_synth  = TransformedGridLoss(use_cuda=True, geometric_model='affine')


```



### Forward Pass

```python


    def forward_base(self, x, contiguous=False, can_detach=True):
        # import pdb; pdb.set_trace()

        # patch feature
        x = x.transpose(1, 2)
        x_pre = self.encoderVideo(x)

        if self.detach_network and can_detach:
            x_pre = x_pre.detach()

        x = self.afterconv1(x_pre)
        x = self.relu(x)

        if contiguous:
            x = x.contiguous()
            x = x.view(x.size(0), x.size(1), x.size(3), x.size(4))

        x_norm = F.normalize(x, p=2, dim=1)

        return x, x_pre, x_norm

    def forward(self, ximg1, patch2, img2, theta):
        B, T = ximg1.shape[:2]
        videoclip1  = ximg1

        # base features
        r50_feat1, r50_feat1_pre, r50_feat1_norm = self.forward_base(videoclip1)

        # target patch feature
        patch2_feat2, patch2_feat2_pre, patch_feat2_norm = self.forward_base(patch2, contiguous=True)

        # target image feature
        img_feat2, img_feat2_pre, img_feat2_norm = self.forward_base(img2, contiguous=True, can_detach=False)


        # base features to crop with transformations
        r50_feat1_transform = r50_feat1.transpose(1, 2)
        channels = r50_feat1_transform.size(2)
        r50_feat1_transform = r50_feat1_transform.contiguous()

        # add original code
        corrfeat1, corrfeat_trans_matrix2, corrfeat_trans1, trans_out2 = self.compute_transform_img_to_patch(patch_feat2_norm, r50_feat1_norm, temporal_out=self.temporal_out)

        bs2 = corrfeat_trans1.size(0)

        r50_feat1_transform_ori = r50_feat1_transform.view(bs2, channels, self.spatial_out1, self.spatial_out1)
        r50_feat1_transform_ori = self.geometricTnf(r50_feat1_transform_ori, trans_out2)

        # r50_feat1_transform_ori = r50_feat1_transform_ori.transpose(1, 2)

        def skip_prediction(img_feat2_norm, r50_feat1_transform_ori):
            r50_feat1_transform_ori = r50_feat1_transform_ori.contiguous()
            r50_feat1_transform_ori = r50_feat1_transform_ori.view(B, self.temporal_out, r50_feat1_transform_ori.size(1),  self.spatial_out2, self.spatial_out2)
            r50_feat1_transform_ori = r50_feat1_transform_ori.transpose(1, 2)

            r50_feat1_transform_norm = F.normalize(r50_feat1_transform_ori, p=2, dim=1)
            corrfeat_trans_matrix_reverse = self.compute_corr_softmax2(img_feat2_norm, r50_feat1_transform_norm)

            corrfeat_trans_reverse  = self.afterconv3_trans(corrfeat_trans_matrix_reverse)
            corrfeat_trans_reverse  = self.leakyrelu(corrfeat_trans_reverse)
            corrfeat_trans_reverse  = self.afterconv4_trans(corrfeat_trans_reverse)
            corrfeat_trans_reverse  = self.leakyrelu(corrfeat_trans_reverse)
            corrfeat_trans_reverse  = corrfeat_trans_reverse.contiguous()
            corrfeat_trans_reverse  = corrfeat_trans_reverse.view(bs2, -1)

            trans_out3  = self.linear2(corrfeat_trans_reverse)
            trans_out3  = trans_out3.contiguous()
            trans_out3  = self.transform_trans_out(trans_out3)

            return trans_out3, corrfeat_trans_matrix_reverse

        trans_out3, corrfeat_trans_matrix_reverse = skip_prediction(img_feat2_norm, r50_feat1_transform_ori)


        def recurrent_align(init_query, idx):
            # global ximg1
            # global patch2

            corr_feat_mats = []
            trans_thetas = []
            trans_feats = []

            # should be normalized patch query
            if self.hist > 1:
                cur_query = torch.stack([init_query]*self.hist)
            else:
                cur_query = init_query

            crops = []
            for t in idx:

                # 1. get affinity of current patch on current frame
                cur_base_norm = r50_feat1_norm[:, :, t:t+1]
                cur_base_feat = r50_feat1_transform[:, t]

                # 2. predict transform with affinity as input
                corrfeat, corrfeat_mat, corrfeat_trans, trans_theta = self.compute_transform_img_to_patch(
                    cur_query if self.hist == 1 else cur_query.mean(0),
                    cur_base_norm)

                # 3. get cropped features with transform
                cur_base_crop = self.geometricTnf(cur_base_feat, trans_theta)

                # bs, channels, time, h2, w2
                cur_base_crop_norm = F.normalize(cur_base_crop, p=2, dim=1)

                # cur_query = cur_base_crop_norm

                if self.hist > 1:
                    cur_query[:-1] = cur_query[1:]
                    cur_query[-1] = cur_base_crop_norm
                else:
                    cur_query = cur_base_crop_norm
                # cur_query = torch.stack([cur_])

                trans_thetas.append(trans_theta)
                trans_feats.append(cur_base_crop)
                # corr_feat_mats.append(corrfeat_mat)

            return trans_thetas, trans_feats #, trans_feats, corr_feat_mats

        def cycle(TT=None):
            if TT is None:
                TT = T

            # propagate backward
            # back_trans_thetas, back_trans_feats, back_corr_feat_mats = \
            back_trans_thetas, back_trans_feats = \
                recurrent_align(patch_feat2_norm, list(range(T))[::-1][:TT])
            # propagate forward
            # forw_trans_thetas, forw_trans_feats, forw_corr_feat_mats = \
            forw_trans_thetas, forw_trans_feats = \
                recurrent_align(F.normalize(back_trans_feats[-1], p=2, dim=1), list(range(T))[T-TT+1:])


            # cycle back from last base frame to target
            last_ = forw_trans_feats[-1] if len(forw_trans_feats) > 0 else back_trans_feats[0]
            last_corrfeat, last_corrfeat_mat, last_corrfeat_trans, last_trans_theta = self.compute_transform_img_to_patch(
                F.normalize(last_, p=2, dim=1), img_feat2_norm.unsqueeze(2))
            last_trans_feat = self.geometricTnf(img_feat2, last_trans_theta)
            last_trans_feat_norm = F.normalize(last_trans_feat, p=2, dim=1)

            forw_trans_thetas.append(last_trans_theta)

            return back_trans_thetas, forw_trans_thetas, back_trans_feats

        # back_trans_thetas, back_trans_feats, back_corr_feat_mats, forw_trans_thetas, forw_trans_feats, forw_corr_feat_mats = \
        #     [], [], [], [], [], []

        outputs = [[], [], []]

        for c in range(1, T+1):
            # _back_trans_thetas, _back_trans_feats, _back_corr_feat_mats, _forw_trans_thetas, _forw_trans_feats, _forw_corr_feat_mats = cycle(c)
            _outputs = cycle(c)
            for i, o in enumerate(_outputs):
                outputs[i] += o

            if c == T:
                back_trans_feats = _outputs[-1]

        back_trans_feats = torch.stack(back_trans_feats).transpose(0,1).contiguous()
        back_trans_feats = back_trans_feats.view(-1, *back_trans_feats.shape[2:])
        skip_trans, skip_corrfeat_mat = skip_prediction(img_feat2_norm, back_trans_feats)

        return outputs[:2], patch2_feat2, theta, trans_out2, trans_out3, skip_trans, skip_corrfeat_mat, corrfeat_trans_matrix2


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







### Correlation Softmax

```python

    def compute_corr_softmax(self, patch_feat1, r50_feat2, detach_corrfeat=False):
        T = r50_feat2.shape[2]

        if detach_corrfeat is True:
            r50_feat2 = r50_feat2.detach()

        r50_feat2 = r50_feat2.transpose(3, 4) # for the inlier counter
        r50_feat2 = r50_feat2.contiguous()
        r50_feat2_vec = r50_feat2.view(r50_feat2.size(0), r50_feat2.size(1), -1)
        r50_feat2_vec = r50_feat2_vec.transpose(1, 2)

        patch_feat1_vec = patch_feat1.view(patch_feat1.size(0), patch_feat1.size(1), -1)
        corrfeat = torch.matmul(r50_feat2_vec, patch_feat1_vec)

        corrfeat = torch.div(corrfeat, self.T)

        corrfeat  = corrfeat.view(corrfeat.size(0), T, self.spatial_out1 * self.spatial_out1, self.spatial_out2, self.spatial_out2)
        corrfeat  = F.softmax(corrfeat, dim=2)
        corrfeat  = corrfeat.view(corrfeat.size(0), T * self.spatial_out1 * self.spatial_out1, self.spatial_out2, self.spatial_out2)

        return corrfeat
    #
    def compute_corr_softmax2(self, patch_feat1, r50_feat2):
        T = r50_feat2.shape[2]

        # bs, channels, time, h2, w2
        r50_feat2 = r50_feat2.contiguous()
        r50_feat2_vec = r50_feat2.view(r50_feat2.size(0), r50_feat2.size(1), -1)
        r50_feat2_vec = r50_feat2_vec.transpose(1, 2)

        # bs, channels, h1, w1
        patch_feat1 = patch_feat1.transpose(2, 3)
        patch_feat1 = patch_feat1.contiguous()
        patch_feat1_vec = patch_feat1.view(patch_feat1.size(0), patch_feat1.size(1), -1)

        corrfeat = torch.matmul(r50_feat2_vec, patch_feat1_vec)
        corrfeat  = torch.div(corrfeat, self.T)
        corrfeat  = corrfeat.contiguous()

        corrfeat  = corrfeat.view(corrfeat.size(0), T, self.spatial_out2 * self.spatial_out2, self.spatial_out1 * self.spatial_out1)
        corrfeat  = F.softmax(corrfeat, dim=3)
        corrfeat  = corrfeat.transpose(2, 3)
        corrfeat  = corrfeat.contiguous()
        corrfeat  = corrfeat.view(corrfeat.size(0) * T, self.spatial_out1 * self.spatial_out1, self.spatial_out2, self.spatial_out2)

        return corrfeat


```



### Loss Functions

```python

    def loss(self, outputs, patch_feat, theta, trans_out2, trans_out3, skip_trans, skip_corrfeat_mat, corrfeat_trans_matrix2):
        # patch_feat is patch of target frame, theta is crop transform for patch

        back_trans_thetas, forw_trans_thetas = outputs

        loss_targ_theta = []
        loss_targ_theta_skip = []
        loss_back_inliers = []

        nn = list(range(len(forw_trans_thetas)))
        nn = [ii for ii in [sum(nn[:i]) - 1 for i in nn][2:] if ii < len(forw_trans_thetas)]

        for i in nn:
            loss_targ_theta.append(self.criterion_synth(forw_trans_thetas[i], theta))

        theta2 = theta.unsqueeze(1)
        theta2 = theta2.repeat(1, self.temporal_out, 1, 1)
        theta2 = theta2.view(-1, 2, 3)

        loss_targ_theta_skip.append(self.criterion_synth(trans_out3, theta2))

        loss_inlier = self.criterion_inlier(matches=corrfeat_trans_matrix2, theta=trans_out2)
        loss_inlier = torch.mean(-loss_inlier)
        loss_back_inliers.append(loss_inlier)

        return (loss_targ_theta, loss_targ_theta_skip, loss_back_inliers)
```





## Related

- [Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://arxivnote.ddlee.cn/2019/09/30/Test-Time-Training-Generalization.html)
- [Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://arxivnote.ddlee.cn/2019/10/11/Temporal-Cycle-Consistency-Learning.html)
- [Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://arxivnote.ddlee.cn/2019/09/30/Test-Time-Training-Generalization.html)
- [Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://arxivnote.ddlee.cn/2019/09/29/Learning-Correspondence-Cycle-Consistency-Time.html)

