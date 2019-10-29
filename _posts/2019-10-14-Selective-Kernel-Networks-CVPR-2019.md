---
title: Selective Kernel Networks
tag:
- CNN
- Image Classification
---

## Info

- Title: Selective Kernel Networks
- Task: Image Classification
- Author: Xiang Li , Wenhai Wang, Xiaolin Hu and Jian Yang
- Date: Mar. 2019
- Arxiv: [1903.06586](https://arxiv.org/abs/1903.06586)
- Published: CVPR 2019



## Highlights

Dynamic sleection mechanism in CNNs that allows each neuron to adaptivel adjust its receptive fied size based on multiple scales of input information.


## Abstract

In standard Convolutional Neural Networks (CNNs), the receptive fields of artificial neurons in each layer are designed to share the same size. It is well-known in the neuroscience community that the receptive field size of visual cortical neurons are modulated by the stimulus, which has been rarely considered in constructing CNNs. We propose a dynamic selection mechanism in CNNs that allows each neuron to adaptively adjust its receptive field size based on multiple scales of input information. A building block called Selective Kernel (SK) unit is designed, in which multiple branches with different kernel sizes are fused using softmax attention that is guided by the information in these branches. Different attentions on these branches yield different sizes of the effective receptive fields of neurons in the fusion layer. Multiple SK units are stacked to a deep network termed Selective Kernel Networks (SKNets). On the ImageNet and CIFAR benchmarks, we empirically show that SKNet outperforms the existing state-of-the-art architectures with lower model complexity. Detailed analyses show that the neurons in SKNet can capture target objects with different scales, which verifies the capability of neurons for adaptively adjusting their receptive field sizes according to the input. The code and models are available at https://github.com/implus/SKNet.



## Motivation & Design

### Selective Kernel

The autorhs  introduce a “Selective Kernel”(SK) convolution, which consists of a triplet of operators: Split, Fuse and Select. The Split operator generates multiple paths with various kernel sizes which correspond to different RF sizes of neurons. The Fuse operator combines
and aggregates the information from multiple paths to obtain a global and comprehensive representation for selection weights. The Select operator aggregates the feature maps of
differently sized kernels according to the selection weights.



![](https://github.com/implus/SKNet/raw/master/figures/sknet.jpg)



## Experiments & Ablation Study

### Single crop validation error on ImageNet-1k (center 224x224/320x320 crop from resized image with shorter side = 256)


| Model | Top-1 224x | Top-1 320x | #P | GFLOPs |
|:-:|:-:|:-:|:-:|:-:|
|ResNeXt-50        |22.23|21.05|25.0M|4.24|
|AttentionNeXt-56  |21.76|–    |31.9M|6.32|
|InceptionV3       |–    |21.20|27.1M|5.73|
|ResNeXt-50 + BAM  |21.70|20.15|25.4M|4.31|
|ResNeXt-50 + CBAM |21.40|20.38|27.7M|4.25|
|SENet-50          |21.12|19.71|27.7M|4.25|
|SKNet-50          |20.79|19.32|27.5M|4.47|
|ResNeXt-101       |21.11|19.86|44.3M|7.99|
|Attention-92      | –   |19.50|51.3M|10.43|
|DPN-92            |20.70|19.30|37.7M|6.50|
|DPN-98            |20.20|18.90|61.6M|11.70|
|InceptionV4       | –   |20.00|42.0M|12.31|
|Inception-ResNetV2| –   |19.90|55.0M|13.22|
|ResNeXt-101 + BAM |20.67|19.15|44.6M|8.05|
|ResNeXt-101 + CBAM|20.60|19.42|49.2M|8.00|
|SENet-101         |20.58|18.61|49.2M|8.00|
|SKNet-101         |20.19|18.40|48.9M|8.46|

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



Average mean attention difference (mean attention value of kernel 5x5 minus that of kernel 3x3) on SK units of SKNet-50, for each of 1,000 categories using all validation samples on ImageNet. On low or middle level SK units (e.g., SK\_2\_3, SK\_3\_4), 5x5 kernels are clearly imposed with more emphasis if the target object becomes larger (1.0x -> 1.5x).

![](https://github.com/implus/SKNet/raw/master/figures/cls_attention_diff.jpg)

More details of attention distributions on specific images:

![](https://github.com/implus/SKNet/raw/master/figures/pics_attention_3_scales.png)



## Code

[Caffe](https://github.com/implus/SKNet)

There are two new layers introduced for efficient training and inference, these are Axpy and CuDNNBatchNorm layers.

### Axpy

Forward

```c
template <typename Dtype>
__global__ void AxpyForward(const int count, const int spatial_dim, 
    const Dtype* scale_data, const Dtype* x_data, const Dtype* y_data,
    Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, count) {
    out_data[index] = scale_data[index / spatial_dim] * x_data[index]
        + y_data[index];
  }
}

template <typename Dtype>
void AxpyLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* scale_data = bottom[0]->gpu_data();
  const Dtype* x_data = bottom[1]->gpu_data();
  const Dtype* y_data = bottom[2]->gpu_data();
  Dtype* out_data = top[0]->mutable_gpu_data();
  const int count = bottom[1]->count();
  AxpyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[1]->count(2), scale_data, x_data, y_data, out_data);  
}

```

Backward
```c
__global__ void AxpyBackwardScale(const int outer_num, const int spatial_dim, 
    const Dtype* x_data, const Dtype* top_diff, Dtype* scale_diff) {
  __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
  unsigned int tid = threadIdx.x;
  buffer[tid] = 0;
  __syncthreads();

  for (int j = tid; j < spatial_dim; j += blockDim.x) {
    int offset = blockIdx.x * spatial_dim + j;
    buffer[tid] += top_diff[offset] * x_data[offset];
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      buffer[threadIdx.x] += buffer[threadIdx.x + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    scale_diff[blockIdx.x] = buffer[0];
  }
}

template <typename Dtype>
__global__ void AxpyBackwardX(const int count, const int spatial_dim, 
    const Dtype* scale_data, const Dtype* top_diff, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = scale_data[index / spatial_dim] * top_diff[index];
  }
}

template <typename Dtype>
void AxpyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (propagate_down[0]) {
    int outer_num = bottom[1]->count(0, 2);
    AxpyBackwardScale<Dtype><<<outer_num, CAFFE_CUDA_NUM_THREADS>>>(
        outer_num, bottom[1]->count(2),
        bottom[1]->gpu_data(), top_diff,
        bottom[0]->mutable_gpu_diff()); 
  }
  if (propagate_down[1]) {
    AxpyBackwardX<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top[0]->count(2), 
        bottom[0]->gpu_data(), top_diff, 
        bottom[1]->mutable_gpu_diff());
  }
  if (propagate_down[2]) {
    caffe_copy(count, top_diff, bottom[2]->mutable_gpu_diff());
  }
  CUDA_POST_KERNEL_CHECK;
}

```

## Related

- [CondConv: Conditionally Parameterized Convolutions for Efficient Inference](https://arxivnote.ddlee.cn/2019/10/15/CondConv-Conditionally-Parameterized-Convolutions-NIPS-2019.html)

- [Deformable Kernels: Adapting Effective Receptive Fields for Object Deformation](https://arxivnote.ddlee.cn/2019/10/13/Deformable-Kernels.html)