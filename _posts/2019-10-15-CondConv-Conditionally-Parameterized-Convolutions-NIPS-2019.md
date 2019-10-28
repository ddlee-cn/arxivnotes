---
title: "CondConv: Conditionally Parameterized Convolutions for Efficient Inference"
tag:
- CNN
---

## Info

- Title: CondConv: Conditionally Parameterized Convolutions for Efficient Inference
- Author: Brandon Yang, Gabriel Bender, Quoc V. Le, Jiquan Ngiam
- Date: Apr. 2019
- Arxiv: [1904.04971](https://arxiv.org/abs/1904.04971)
- Published: NIPS 2019


## Abstract

Convolutional layers are one of the basic building blocks of modern deep neural networks. One fundamental assumption is that convolutional kernels should be shared for all examples in a dataset. We propose conditionally parameterized convolutions (CondConv), which learn specialized convolutional kernels for each example. Replacing normal convolutions with CondConv enables us to increase the size and capacity of a network, while maintaining efficient inference. We demonstrate that scaling networks with CondConv improves the performance and inference cost trade-off of several existing convolutional neural network architectures on both classification and detection tasks. On ImageNet classification, our CondConv approach applied to EfficientNet-B0 achieves state-of-the-art performance of 78.3% accuracy with only 413M multiply-adds. 



## Motivation & Design

Conditionally parameterized convolutions (CondConv) are a new building block for convolutional neural networks to increase capacity while maintaining efficient inference. In a traditional convolutional layer, each example is processed with the same kernel. In a CondConv layer, each example is processed with a specialized, example-dependent kernel. As an intuitive motivating example, on the ImageNet classification dataset, we might want to classify dogs and cats with different convolutional kernels.



![](https://github.com/tensorflow/tpu/raw/master/models/official/efficientnet/g3doc/condconv-layer.png)

$$
\text { Output }(x)=\sigma\left(\left(\alpha_{1} \cdot W_{1}+\ldots+\alpha_{n} \cdot W_{n}\right) * x\right)
$$

where each $\alpha_i = r_i(x)$ is an example-dependent scalar weight computed using a routing function with learned parameters, $n$ is the number of experts, and $σ$ is an activation function. When we adapt a convolutional layer to use CondConv, each kernel Wi has the same dimensions as the kernel in the original convolution.


A CondConv layer consists of n experts, each of which are the same size as the convolutional kernel of the original convolutional layer. For each example, the example-dependent convolutional kernel is computed as the weighted sum of experts using an example-dependent routing function. Increasing the number of experts enables us to increase the capacity of a network, while maintaining efficient inference.

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


## Code


[TensorFlow](<https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/condconv/>)


call with external `routing weights` and built-in `condconv_kernels`

```python
class CondConv2D(tf.keras.layers.Conv2D):
  def __init__(self,
               filters,
               kernel_size,
               num_experts,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(CondConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    if num_experts < 1:
      raise ValueError('A CondConv layer must have at least one expert.')
    self.num_experts = num_experts
    if self.data_format == 'channels_first':
      self.converted_data_format = 'NCHW'
    else:
      self.converted_data_format = 'NHWC'

  def build(self, input_shape):
    if len(input_shape) != 4:
      raise ValueError(
          'Inputs to `CondConv2D` should have rank 4. '
          'Received input shape:', str(input_shape))
    input_shape = tf.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])

    self.kernel_shape = self.kernel_size + (input_dim, self.filters)
    kernel_num_params = 1
    for kernel_dim in self.kernel_shape:
      kernel_num_params *= kernel_dim
    condconv_kernel_shape = (self.num_experts, kernel_num_params)
    self.condconv_kernel = self.add_weight(
        name='condconv_kernel',
        shape=condconv_kernel_shape,
        initializer=get_condconv_initializer(self.kernel_initializer,
                                             self.num_experts,
                                             self.kernel_shape),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

    if self.use_bias:
      self.bias_shape = (self.filters,)
      condconv_bias_shape = (self.num_experts, self.filters)
      self.condconv_bias = self.add_weight(
          name='condconv_bias',
          shape=condconv_bias_shape,
          initializer=get_condconv_initializer(self.bias_initializer,
                                               self.num_experts,
                                               self.bias_shape),
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

    self.input_spec = tf.layers.InputSpec(
        ndim=self.rank + 2, axes={channel_axis: input_dim})

    self.built = True

  def call(self, inputs, routing_weights):
    # Compute example dependent kernels
    kernels = tf.matmul(routing_weights, self.condconv_kernel)
    batch_size = inputs.shape[0].value
    inputs = tf.split(inputs, batch_size, 0)
    kernels = tf.split(kernels, batch_size, 0)
    # Apply example-dependent convolution to each example in the batch
    outputs_list = []
    for input_tensor, kernel in zip(inputs, kernels):
      kernel = tf.reshape(kernel, self.kernel_shape)
      outputs_list.append(
          tf.nn.convolution(
              input_tensor,
              kernel,
              strides=self.strides,
              padding=self._get_padding_op(),
              dilations=self.dilation_rate,
              data_format=self.converted_data_format))
    outputs = tf.concat(outputs_list, 0)

    if self.use_bias:
      # Compute example-dependent biases
      biases = tf.matmul(routing_weights, self.condconv_bias)
      outputs = tf.split(outputs, batch_size, 0)
      biases = tf.split(biases, batch_size, 0)
      # Add example-dependent bias to each example in the batch
      bias_outputs_list = []
      for output, bias in zip(outputs, biases):
        bias = tf.squeeze(bias, axis=0)
        bias_outputs_list.append(
            tf.nn.bias_add(output, bias,
                           data_format=self.converted_data_format))
      outputs = tf.concat(bias_outputs_list, 0)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs
```


```python
class DepthwiseCondConv2D(tf.keras.layers.DepthwiseConv2D):
  def __init__(self,
               kernel_size,
               num_experts,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DepthwiseCondConv2D, self).__init__(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    if num_experts < 1:
      raise ValueError('A CondConv layer must have at least one expert.')
    self.num_experts = num_experts
    if self.data_format == 'channels_first':
      self.converted_data_format = 'NCHW'
    else:
      self.converted_data_format = 'NHWC'

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError(
          'Inputs to `DepthwiseCondConv2D` should have rank 4. '
          'Received input shape:', str(input_shape))
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = 3
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                                   input_dim, self.depth_multiplier)

    depthwise_kernel_num_params = 1
    for dim in self.depthwise_kernel_shape:
      depthwise_kernel_num_params *= dim
    depthwise_condconv_kernel_shape = (self.num_experts,
                                       depthwise_kernel_num_params)

    self.depthwise_condconv_kernel = self.add_weight(
        shape=depthwise_condconv_kernel_shape,
        initializer=get_condconv_initializer(self.depthwise_initializer,
                                             self.num_experts,
                                             self.depthwise_kernel_shape),
        name='depthwise_condconv_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint,
        trainable=True)

    if self.use_bias:
      bias_dim = input_dim * self.depth_multiplier
      self.bias_shape = (bias_dim,)
      condconv_bias_shape = (self.num_experts, bias_dim)
      self.condconv_bias = self.add_weight(
          name='condconv_bias',
          shape=condconv_bias_shape,
          initializer=get_condconv_initializer(self.bias_initializer,
                                               self.num_experts,
                                               self.bias_shape),
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = tf.layers.InputSpec(
        ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs, routing_weights):
    # Compute example dependent depthwise kernels
    depthwise_kernels = tf.matmul(routing_weights,
                                  self.depthwise_condconv_kernel)
    batch_size = inputs.shape[0].value
    inputs = tf.split(inputs, batch_size, 0)
    depthwise_kernels = tf.split(depthwise_kernels, batch_size, 0)
    # Apply example-dependent depthwise convolution to each example in the batch
    outputs_list = []
    for input_tensor, depthwise_kernel in zip(inputs, depthwise_kernels):
      depthwise_kernel = tf.reshape(depthwise_kernel,
                                    self.depthwise_kernel_shape)
      if self.data_format == 'channels_first':
        converted_strides = (1, 1) + self.strides
      else:
        converted_strides = (1,) + self.strides + (1,)
      outputs_list.append(
          tf.nn.depthwise_conv2d(
              input_tensor,
              depthwise_kernel,
              strides=converted_strides,
              padding=self.padding.upper(),
              dilations=self.dilation_rate,
              data_format=self.converted_data_format))
    outputs = tf.concat(outputs_list, 0)

    if self.use_bias:
      # Compute example-dependent biases
      biases = tf.matmul(routing_weights, self.condconv_bias)
      outputs = tf.split(outputs, batch_size, 0)
      biases = tf.split(biases, batch_size, 0)
      # Add example-dependent bias to each example in the batch
      bias_outputs_list = []
      for output, bias in zip(outputs, biases):
        bias = tf.squeeze(bias, axis=0)
        bias_outputs_list.append(
            tf.nn.bias_add(output, bias,
                           data_format=self.converted_data_format))
      outputs = tf.concat(bias_outputs_list, 0)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs
```

## Related

