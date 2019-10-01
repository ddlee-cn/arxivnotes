---
title: (VQ-VAE)Neural Discrete Representation Learning - van den Oord - NIPS 2017 - TensorFlow & PyTorch Code
tag:
- VAE
- Image Generation
redirect_from: /Neural-Discrete-Representation-Learning.html
---



## Info
- Title: **Neural Discrete Representation Learning**
- Task: **Image Generation**
- Author: A. van den Oord, O. Vinyals, and K. Kavukcuoglu
- Date: Nov. 2017
- Arxiv: [1711.00937](https://arxiv.org/abs/1711.00937)
- Published: NIPS 2017
- Affiliation: Google DeepMind


## Highlights & Drawbacks
- Discrete representation for data distribution
- The prior is learned instead of random

## Motivation & Design

### Vector Quantisation(VQ)
Vector quantisation (VQ) is a method to map $K$-dimensional vectors into a finite set of “code” vectors. The encoder output $E(\mathbf{x})=\mathbf{z}_{e}$ goes through a nearest-neighbor lookup to match to one of $K$ embedding vectors and then this matched code vector becomes the input for the decoder $D(.)$:

$$
z_{q}(x)=e_{k}, \quad \text { where } \quad k=\operatorname{argmin}_{j}\left\|z_{e}(x)-e_{j}\right\|_{2}
$$

The dictionary items are updated using Exponential Moving Averages(EMA), which is similar to EM methods like K-Means.

![(VQ-VAE)Neural Discrete Representation Learning](https://i.imgur.com/O8c2e05.png)


### Loss Design
- Reconstruction loss
- VQ loss: The L2 error between the embedding space and the encoder outputs.
- Commitment loss: A measure to encourage the encoder output to stay close to the embedding space and to prevent it from fluctuating too frequently from one code vector to another.

$$
L=\underbrace{\left\|\mathbf{x}-D\left(\mathbf{e}_{k}\right)\right\|_{2}^{2}}_{\text { reconstruction loss }}+\underbrace{\left\|\operatorname{sg}[E(\mathbf{x})]-\mathbf{e}_{k}\right\|_{2}^{2}}_{\text { VQ loss }}+\underbrace{\beta\left\|E(\mathbf{x})-\operatorname{sg}\left[\mathbf{e}_{k}\right]\right\|_{2}^{2}}_{\text { commitment loss }}
$$

where sq[.] is the  `stop_gradient`  operator.

### Prior
Training PixelCNN and WaveNet for images and audio respectively on learned latent space, the VA-VAE model avoids "posterior collapse" problem which VAE suffers from.


## Performance & Ablation Study

Original VS. Reconstruction:
![(VQ-VAE)Neural Discrete Representation Learning](https://i.imgur.com/oUgQSUz.png)


Category Image Generation:
![(VQ-VAE)Neural Discrete Representation Learning](https://i.imgur.com/k3CzNQl.png)



## Code
- [PyTorch](https://github.com/zalandoresearch/pytorch-vq-vae)
- [Sonnet Implementation](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py)
- [Sonnet Example](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb)
- [Project Site](https://avdnoord.github.io/homepage/vqvae/)

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


### Vector Quantization

```python
class VectorQuantizer(base.AbstractModule):
  """Sonnet module representing the VQ-VAE layer.
  Implements the algorithm presented in
  'Neural Discrete Representation Learning' by van den Oord et al.
  https://arxiv.org/abs/1711.00937
  Input any tensor to be quantized. Last dimension will be used as space in
  which to quantize. All other dimensions will be flattened and will be seen
  as different examples to quantize.
  The output tensor will have the same shape as the input.
  For example a tensor with shape [16, 32, 32, 64] will be reshaped into
  [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
  independently.
  Args:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms
      (see equation 4 in the paper - this variable is Beta).
  """

  def __init__(self, embedding_dim, num_embeddings, commitment_cost,
               name='vq_layer'):
    super(VectorQuantizer, self).__init__(name=name)
    self._embedding_dim = embedding_dim
    self._num_embeddings = num_embeddings
    self._commitment_cost = commitment_cost

    with self._enter_variable_scope():
      initializer = tf.uniform_unit_scaling_initializer()
      self._w = tf.get_variable('embedding', [embedding_dim, num_embeddings],
                                initializer=initializer, trainable=True)

  def _build(self, inputs, is_training):
    """Connects the module to some inputs.
    Args:
      inputs: Tensor, final dimension must be equal to embedding_dim. All other
        leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data.
    Returns:
      dict containing the following keys and values:
        quantize: Tensor containing the quantized version of the input.
        loss: Tensor containing the loss to optimize.
        perplexity: Tensor containing the perplexity of the encodings.
        encodings: Tensor containing the discrete encodings, ie which element
          of the quantized space each input element was mapped to.
        encoding_indices: Tensor containing the discrete encoding indices, ie
          which element of the quantized space each input element was mapped to.
    """
    # Assert last dimension is same as self._embedding_dim
    input_shape = tf.shape(inputs)
    with tf.control_dependencies([
        tf.Assert(tf.equal(input_shape[-1], self._embedding_dim),
                  [input_shape])]):
      flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

    distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                 - 2 * tf.matmul(flat_inputs, self._w)
                 + tf.reduce_sum(self._w ** 2, 0, keepdims=True))

    encoding_indices = tf.argmax(- distances, 1)
    encodings = tf.one_hot(encoding_indices, self._num_embeddings)
    encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)

    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
    loss = q_latent_loss + self._commitment_cost * e_latent_loss

    quantized = inputs + tf.stop_gradient(quantized - inputs)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))

    return {'quantize': quantized,
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings,
            'encoding_indices': encoding_indices,}

  @property
  def embeddings(self):
    return self._w

  def quantize(self, encoding_indices):
    with tf.control_dependencies([encoding_indices]):
      w = tf.transpose(self.embeddings.read_value(), [1, 0])
    return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

```





## Related

- VQ-VAE-2: [ Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019](https://arxivnote.ddlee.cn/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2.html)
- [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modification - Salimans - ICLR 2017](https://arxivnote.ddlee.cn/PixelCNN++-Improving-the-PixelCNN-with-Discretized-Logistic-Mixture-Likelihood-and-Other-Modification.html)
- Gated PixelCNN: [Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016](https://arxivnote.ddlee.cn/Conditional-Image-Generation-with-PixelCNN-Decoders.html)
- PixelRNN & PixelCNN: [Pixel Recurrent Neural Networks - van den Oord - ICML 2016](https://arxivnote.ddlee.cn/Pixel-Recurrent-Neural-Networks.html)