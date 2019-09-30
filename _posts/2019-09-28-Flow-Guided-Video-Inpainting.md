---
title: Deep Flow-Guided Video Inpainting - Rui Xu - CVPR 2019
tag:
- Video Inpainting
- Video
- Low-Level Vision

---

## Info

- Title: Deep Flow-Guided Video Inpainting
- Task: Video Inpainting
- Author: Rui Xu, Xiaoxiao Li, Bolei Zhou, Chen Change Loy
- Date: May 2019
- Arxiv: [1905.02884](https://arxiv.org/abs/1905.02884)
- Published: CVPR 2019




## Abstract

Video inpainting, which aims at filling in missing regions of a video, remains challenging due to the difficulty of preserving the precise spatial and temporal coherence of video contents. In this work we propose a novel flow-guided video inpainting approach. Rather than filling in the RGB pixels of each frame directly, we consider video inpainting as a pixel propagation problem. We first synthesize a spatially and temporally coherent optical flow field across video frames using a newly designed Deep Flow Completion network. Then the synthesized flow field is used to guide the propagation of pixels to fill up the missing regions in the video. Specifically, the Deep Flow Completion network follows a coarse-to-fine refinement to complete the flow fields, while their quality is further improved by hard flow example mining. Following the guide of the completed flow, the missing video regions can be filled up precisely. Our method is evaluated on DAVIS and YouTube-VOS datasets qualitatively and quantitatively, achieving the state-of-the-art performance in terms of inpainting quality and speed.



## Motivation & Design

The framework contains two steps, the first step is to complete the missing flow while the second step is to propagate pixels with the guidance of completed flow fields. In the first step, a Deep Flow Completion Network (DFC-Net) is proposed for coarse-to-fine flow completion. DFC-Net consists of three similar subnetworks named as DFC-S. The first subnetwork estimates the flow in a relatively coarse scale and feeds them into the second and third subnetwork for further refinement. In the second step, after the flow is obtained, most of the missing regions can be filled up by pixels in known regions through a flow-guided propagation from different frames. A conventional image inpainting network is finally employed to complete the remaining regions that are not seen in the entire video. Thanks to the high-quality estimated flow in the first step, we can easily propagate these image inpainting results to the entire video sequence.


![Flow-Guided Video Inpainting](https://nbei.github.io/video-inpainting/framework.png)

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


(1) Coarse-to-fine refinement: The proposed DFC-Net is designed to recover accurate flow field from missing regions. This is made possible through stacking three similar subnetworks (DFC-S) to perform coarse-to-fine flow completion. Specifically, the first subnetwork accepts a batch of consecutive frames as the input and estimates the missing flow of the middle frame on a relatively coarse scale. The batch of coarsely estimated flow fields is subsequently fed to the second subnetwork followed by the third subnetwork for further spatial resolution and accuracy refinement.

(2) Temporal coherence maintenance: Our DFC-Net is designed to naturally encourage global temporal consistency even though its subnetworks only predict a single frame each time. This is achieved through feeding a batch of consecutive frames as inputs, which provide richer temporal information. In addition, the highly similar inputs between adjacent frames tend to produce continuous results.

(3) Hard flow example mining: We introduce hard flow example mining strategy to improve the inpainting quality on flow boundary and dynamic regions.



## Experiments & Ablation Study

For each input sequence (odd row), we show representative frames with mask of missing region overlay. We show the inpainting results in even rows.

![Flow-Guided Video Inpainting](https://nbei.github.io/video-inpainting/final.png)



## Code

[Project Site](https://nbei.github.io/video-inpainting.html)

[PyTorch](https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting)



### The Coarse-to-fine Generator(DeepFill V1 Model)

```python
class Generator(nn.Module):
    def __init__(self, first_dim=32, isCheck=False, device=None):
        super(Generator, self).__init__()
        self.isCheck = isCheck
        self.device = device
        self.stage_1 = CoarseNet(5, first_dim, device=device)
        self.stage_2 = RefinementNet(5, first_dim, device=device)

    def forward(self, masked_img, mask, small_mask): # mask : 1 x 1 x H x W
        # border, maybe
        mask = mask.expand(masked_img.size(0),1,masked_img.size(2),masked_img.size(3))
        small_mask = small_mask.expand(masked_img.size(0), 1, masked_img.size(2) // 8, masked_img.size(3) // 8)
        if self.device:
            ones = to_var(torch.ones(mask.size()), device=self.device)
        else:
            ones = to_var(torch.ones(mask.size()))
        # stage1
        stage1_input = torch.cat([masked_img, ones, ones*mask], dim=1)
        stage1_output, resized_mask = self.stage_1(stage1_input, mask)
        # stage2
        new_masked_img = stage1_output*mask.clone() + masked_img.clone()*(1.-mask.clone())
        stage2_input = torch.cat([new_masked_img, ones.clone(), ones.clone()*mask.clone()], dim=1)
        stage2_output, offset_flow = self.stage_2(stage2_input, small_mask)

        return stage1_output, stage2_output, offset_flow


class CoarseNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''
    def __init__(self, in_ch, out_ch, device=None):
        super(CoarseNet,self).__init__()
        self.down = Down_Module(in_ch, out_ch)
        self.atrous = Dilation_Module(out_ch*4, out_ch*4)
        self.up = Up_Module(out_ch*4, 3)
        self.device=device

    def forward(self, x, mask):
        x = self.down(x)
        resized_mask = down_sample(mask, scale_factor=0.25, mode='nearest', device=self.device)
        x = self.atrous(x)
        x = self.up(x)

        return x, resized_mask


class RefinementNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''
    def __init__(self, in_ch, out_ch, device=None):
        super(RefinementNet,self).__init__()
        self.down_conv_branch = Down_Module(in_ch, out_ch, isRefine=True)
        self.down_attn_branch = Down_Module(in_ch, out_ch, activation=nn.ReLU(), isRefine=True, isAttn=True)
        self.atrous = Dilation_Module(out_ch*4, out_ch*4)
        self.CAttn = Contextual_Attention_Module(out_ch*4, out_ch*4, device=device)
        self.up = Up_Module(out_ch*8, 3, isRefine=True)

    def forward(self, x, resized_mask):
        # conv branch
        conv_x = self.down_conv_branch(x)
        conv_x = self.atrous(conv_x)

        # attention branch
        attn_x = self.down_attn_branch(x)

        attn_x, offset_flow = self.CAttn(attn_x, attn_x, mask=resized_mask)

        # concat two branches
        deconv_x = torch.cat([conv_x, attn_x], dim=1) # deconv_x => B x 256 x W/4 x H/4
        x = self.up(deconv_x)

        return x, offset_flow
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

### Bidirectional Propagation Process

```python
while masked_frame_num > 0:
    
    results = [
        np.zeros(image.shape + (2,), dtype=image.dtype)
        for _ in range(frames_num)
    ]
    time_stamp = [
        -np.ones(image.shape[:2] + (2,), dtype=int)
        for _ in range(frames_num)
    ]

    # forward
    if iter_num == 0:
        image = cv2.imread(os.path.join(img_root, frame_name_list[0]))
        image = cv2.resize(image, (shape[1], shape[0]))
        if args.FIX_MASK:
            label = cv2.imread(
                os.path.join(mask_root), cv2.IMREAD_UNCHANGED)
        else:
            label = cv2.imread(
                os.path.join(mask_root, '%05d.png' % (0 + flow_start_no)), cv2.IMREAD_UNCHANGED)
        label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        image = result_pool[0]
        label = label_pool[0]

    if len(label.shape) == 3:
        label = label[:, :, 0]
    if args.enlarge_mask and iter_num == 0:
        kernel = np.ones((args.enlarge_kernel, args.enlarge_kernel), np.uint8)
        label = cv2.dilate(label, kernel, iterations=1)

    label = (label > 0).astype(np.uint8)
    image[label > 0, :] = 0

    results[0][..., 0] = image
    time_stamp[0][label == 0, 0] = 0
    for th in range(1, frames_num):
        prog_bar.update()
        if iter_num == 0:
            image = cv2.imread(os.path.join(img_root, frame_name_list[th]))
            image = cv2.resize(image, (shape[1], shape[0]))
        else:
            image = result_pool[th]

        flow1 = flo.readFlow(os.path.join(flow_root, '%05d.flo' % (th - 1 + flow_start_no)))
        flow2 = flo.readFlow(os.path.join(flow_root, '%05d.rflo' % (th + flow_start_no)))
        flow1 = flo.flow_tf(flow1, image.shape)
        flow2 = flo.flow_tf(flow2, image.shape)

        if iter_num == 0:
            if not args.FIX_MASK:
                label = cv2.imread(
                    os.path.join(mask_root, '%05d.png' % (th + flow_start_no)), cv2.IMREAD_UNCHANGED)
            else:
                label = cv2.imread(
                    os.path.join(mask_root), cv2.IMREAD_UNCHANGED)
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            label = label_pool[th]

        if len(label.shape) == 3:
            label = label[:, :, 0]

        if args.enlarge_mask and iter_num == 0:
            kernel = np.ones((args.enlarge_kernel, args.enlarge_kernel), np.uint8)
            label = cv2.dilate(label, kernel, iterations=1)

        label = (label > 0).astype(np.uint8)
        image[(label > 0), :] = 0

        temp1 = flo.get_warp_label(flow1, flow2,
                                   results[th - 1][..., 0],
                                   th=th_warp)
        temp2 = flo.get_warp_label(flow1, flow2,
                                   time_stamp[th - 1],
                                   th=th_warp,
                                   value=-1)[..., 0]

        results[th][..., 0] = temp1
        time_stamp[th][..., 0] = temp2

        results[th][label == 0, :, 0] = image[label == 0, :]
        time_stamp[th][label == 0, 0] = th

    # backward
    if iter_num == 0:

        image = cv2.imread(
            os.path.join(img_root, frame_name_list[frames_num - 1]))
        image = cv2.resize(image, (shape[1], shape[0]))

        if not args.FIX_MASK:
            label = cv2.imread(
                os.path.join(mask_root, '%05d.png' % (frames_num - 1 + flow_start_no)),
                cv2.IMREAD_UNCHANGED)
        else:
            label = cv2.imread(
                os.path.join(mask_root),
                cv2.IMREAD_UNCHANGED)
        label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        image = result_pool[-1]
        label = label_pool[-1]

    if len(label.shape) == 3:
        label = label[:, :, 0]
    if args.enlarge_mask and iter_num == 0:
        kernel = np.ones((args.enlarge_kernel, args.enlarge_kernel), np.uint8)
        label = cv2.dilate(label, kernel, iterations=1)

    label = (label > 0).astype(np.uint8)
    image[(label > 0), :] = 0

    results[frames_num - 1][..., 1] = image
    time_stamp[frames_num - 1][label == 0, 1] = frames_num - 1
    prog_bar = ProgressBar(frames_num-1)
    for th in range(frames_num - 2, -1, -1):
        prog_bar.update()
        if iter_num == 0:
            image = cv2.imread(os.path.join(img_root, frame_name_list[th]))
            image = cv2.resize(image, (shape[1], shape[0]))
            if not args.FIX_MASK:
                label = cv2.imread(
                    os.path.join(mask_root, '%05d.png' % (th + flow_start_no)), cv2.IMREAD_UNCHANGED)
            else:
                label = cv2.imread(
                    os.path.join(mask_root), cv2.IMREAD_UNCHANGED)
            label = cv2.resize(label, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            image = result_pool[th]
            label = label_pool[th]

        flow1 = flo.readFlow(os.path.join(flow_root, '%05d.rflo' % (th + 1 + flow_start_no)))
        flow2 = flo.readFlow(os.path.join(flow_root, '%05d.flo' % (th + flow_start_no)))
        flow1 = flo.flow_tf(flow1, image.shape)
        flow2 = flo.flow_tf(flow2, image.shape)

        if len(label.shape) == 3:
            label = label[:, :, 0]
        if args.enlarge_mask and iter_num == 0:
            kernel = np.ones((args.enlarge_kernel, args.enlarge_kernel), np.uint8)
            label = cv2.dilate(label, kernel, iterations=1)

        label = (label > 0).astype(np.uint8)
        image[(label > 0), :] = 0

        temp1 = flo.get_warp_label(flow1, flow2,
                                   results[th + 1][..., 1],
                                   th=th_warp)
        temp2 = flo.get_warp_label(
            flow1, flow2, time_stamp[th + 1],
            value=-1,
            th=th_warp,
        )[..., 1]

        results[th][..., 1] = temp1
        time_stamp[th][..., 1] = temp2

        results[th][label == 0, :, 1] = image[label == 0, :]
        time_stamp[th][label == 0, 1] = th

    sys.stdout.write('\n')
    tmp_label_seq = np.zeros(frames_num-1)

    # merge
    prog_bar = ProgressBar(frames_num)
    for th in range(0, frames_num - 1):
        prog_bar.update()
        v1 = (time_stamp[th][..., 0] == -1)
        v2 = (time_stamp[th][..., 1] == -1)

        hole_v = (v1 & v2)

        result = results[th][..., 0].copy()
        result[v1, :] = results[th][v1, :, 1].copy()

        v3 = ((v1 == 0) & (v2 == 0))

        dist = time_stamp[th][..., 1] - time_stamp[th][..., 0]
        dist[dist < 1] = 1

        w2 = (th - time_stamp[th][..., 0]) / dist
        w2 = (w2 > 0.5).astype(np.float)

        result[v3, :] = (results[th][..., 1] * w2[..., np.newaxis] +
                         results[th][..., 0] * (1 - w2)[..., np.newaxis])[v3, :]

        result_pool[th] = result.copy()

        tmp_mask = np.zeros_like(result)
        tmp_mask[hole_v, :] = 255
        label_pool[th] = tmp_mask.copy()
        tmp_label_seq[th] = np.sum(tmp_mask)

    sys.stdout.write('\n')
    frame_inpaint_seq[tmp_label_seq == 0] = 0
    masked_frame_num = np.sum((frame_inpaint_seq > 0).astype(np.int))
    print(masked_frame_num)
    iter_num += 1

    if masked_frame_num > 0:
        key_frame_ids = get_key_ids(frame_inpaint_seq)
        print(key_frame_ids)
        for id in key_frame_ids:
            with torch.no_grad():
                tmp_inpaint_res = frame_inapint_model.forward(result_pool[id], label_pool[id])
            label_pool[id] = label_pool[id] * 0.
            result_pool[id] = tmp_inpaint_res
    else:
        print(frames_num, 'frames have been inpainted by', iter_num, 'iterations.')

    tmp_label_seq = np.zeros(frames_num - 1)
    for th in range(0, frames_num - 1):
        tmp_label_seq[th] = np.sum(label_pool[th])
    frame_inpaint_seq[tmp_label_seq == 0] = 0
    masked_frame_num = np.sum((frame_inpaint_seq > 0).astype(np.int))

```



## Related

- [An Internal Learning Approach to Video Inpainting - ICCV 2019](https://arxivnote.ddlee.cn/2019/09/24/Internal-Learning-Video-Inpainting-ICCV.html)
- [Image Inpainting: From PatchMatch to Pluralistic](https://arxivnote.ddlee.cn/2019/09/22/Image-Inpainting-PatchMatch-Edge-Connect-Partial-Conv.html)
- [Deep Image Prior - Ulyanov - CVPR 2018](https://arxivnote.ddlee.cn/2019/08/26/Deep-Image-Prior-Ulyanov-CVPR-2018.html)
- [Generative Image Inpainting with Contextual Attention - Yu - CVPR 2018 - TensorFlow](https://arxivnote.ddlee.cn/2019/08/06/Generative-Image-Inpainting-with-Contextual-Attention-Yu-CVPR-TensorFlow.html)
- [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning - Nazeri - 2019 - PyTorch](https://arxivnote.ddlee.cn/2019/08/05/EdgeConnect-Generative-Image-Inpainting-with-Adversarial-Edge-Learning-Nazeri.html)
- [Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://arxivnote.ddlee.cn/2019/08/04/Globally-and-locally-consistent-image-completion-SIGGRAPH.html)