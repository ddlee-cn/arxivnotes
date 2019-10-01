---
title: "Glow: Generative Flow with Invertible 1x1 Convolutions - Kingma & Dhariwal - NIPS 2018"
tag:
- AutoRegressive
- Image Generation
redirect_from: /Glow-Generative-Flow-with-Invertible-1x1-Convolutions-Kingma-Dhariwal-NIPS.html
---



## Info
- Title: **Glow: Generative Flow with Invertible 1x1 Convolutions**
- Task: **Image Generation**
- Author: D. P. Kingma and P. Dhariwal
- Date: Jul. 2018
- Arxiv: [1807.03039](https://arxiv.org/abs/1807.03039)
- Published: NIPS 2018

## Motivation & Design
### The merits of flow-based models

- Exact latent-variable inference and log-likelihood evaluation. In VAEs, one is able to infer only approximately the value of the latent variables that correspond to a datapoint. GAN’s have no encoder at all to infer the latents. In reversible generative models, this can be done exactly without approximation. Not only does this lead to accurate inference, it also enables optimization of the exact log-likelihood of the data, instead of a lower bound of it.
- Efficient inference and efficient synthesis. Autoregressive models, such as the Pixel- CNN (van den Oord et al., 2016b), are also reversible, however synthesis from such models is difficult to parallelize, and typically inefficient on parallel hardware. Flow-based generative models like Glow (and RealNVP) are efficient to parallelize for both inference and synthesis.
- Useful latent space for downstream tasks. The hidden layers of autoregressive models have unknown marginal distributions, making it much more difficult to perform valid manipulation of data. In GANs, data points can usually not be directly represented in a latent space, as they have no encoder and might not have full support over the data distribution. (Grover et al., 2018). This is not the case for reversible generative models and VAEs, which allow for various applications such as interpolations between data points and meaningful modifications of existing data points.
- Significant potential for memory savings. Computing gradients in reversible neural networks requires an amount of memory that is constant instead of linear in their depth.

### The proposed flow
![CleanShot 2019-08-20 at 15.25.05@2x](https://i.imgur.com/L797JWn.jpg)

The authors propose a generative flow where each step (left) consists of an actnorm step, followed by an invertible 1 × 1 convolution, followed by an affine transformation (Dinh et al., 2014). This flow is combined with a multi-scale architecture (right).

There are three steps in one stage of flow in Glow.

Step 1:**Activation normalization** (short for “actnorm”)

It performs an affine transformation using a scale and bias parameter per channel, similar to batch normalization, but works for mini-batch size 1. The parameters are trainable but initialized so that the first minibatch of data have mean 0 and standard deviation 1 after actnorm.

Step 2: **Invertible 1x1 conv**

Between layers of the RealNVP flow, the ordering of channels is reversed so that all the data dimensions have a chance to be altered. A 1×1 convolution with equal number of input and output channels is a generalization of any permutation of the channel ordering.

Say, we have an invertible 1x1 convolution of an input $h×w×c$ tensor $h$ with a weight matrix $W$ of size $c×c$. The output is a $h×w×c$ tensor, labeled as $ f=𝚌𝚘𝚗𝚟𝟸𝚍(h;W)$. In order to apply the change of variable rule, we need to compute the Jacobian determinant $|det∂f/∂h|$.

Both the input and output of 1x1 convolution here can be viewed as a matrix of size $h×w$. Each entry $x_{ij}$($i=1,2...h, j=1,2,...,w$) in $h$ is a vector of $c$ channels and each entry is multiplied by the weight matrix $W$ to obtain the corresponding entry $y_{ij}$ in the output matrix respectively. The derivative of each entry is $\partial \mathbf{x}_{i j} \mathbf{W} / \partial \mathbf{x}_{i j}=\mathbf{w}$ and there are $h×w$ such entries in total:

The inverse 1x1 convolution depends on the inverse matrix $W^{−1}$
. Since the weight matrix is relatively small, the amount of computation for the matrix determinant (tf.linalg.det) and inversion (tf.linalg.inv) is still under control.

Step 3: Affine coupling layer

The design is same as in RealNVP.

![CleanShot 2019-08-20 at 11.17.30@2x](https://i.imgur.com/kqWmlSn.jpg)

The three main components of proposed flow, their reverses, and their log-determinants. Here, $x$ signifies the input of the layer, and $y$ signifies its output. Both $x$ and $y$ are tensors of shape $[h × w × c]$ with spatial dimensions (h, w) and channel dimension $c$. With $(i, j)$ we denote spatial indices into tensors $x$ and $y$. The function NN() is a nonlinear mapping, such as a (shallow) convolutional neural network like in ResNets (He et al., 2016) and RealNVP (Dinh et al., 2016).


## Performance & Ablation Study
![CleanShot 2019-08-20 at 11.18.32@2x](https://i.imgur.com/NEX73i5.jpg)

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
- [Blog Post](https://openai.com/blog/glow)
- [TensorFlow](https://github.com/openai/glow)

### Model

```python
import tfops as Z

def model(sess, hps, train_iterator, test_iterator, data_init):

    # Only for decoding/init, rest use iterators directly
    with tf.name_scope('input'):
        X = tf.placeholder(
            tf.uint8, [None, hps.image_size, hps.image_size, 3], name='image')
        Y = tf.placeholder(tf.int32, [None], name='label')
        lr = tf.placeholder(tf.float32, None, name='learning_rate')

    encoder, decoder = codec(hps)
    hps.n_bins = 2. ** hps.n_bits_x

    def _f_loss(x, y, is_training, reuse=False):

        with tf.variable_scope('model', reuse=reuse):
            y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')

            # Discrete -> Continuous
            objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]
            z = preprocess(x)
            z = z + tf.random_uniform(tf.shape(z), 0, 1./hps.n_bins)
            objective += - np.log(hps.n_bins) * np.prod(Z.int_shape(z)[1:])

            # Encode
            z = Z.squeeze2d(z, 2)  # > 16x16x12
            z, objective, _ = encoder(z, objective)

            # Prior
            hps.top_shape = Z.int_shape(z)[1:]
            logp, _, _ = prior("prior", y_onehot, hps)
            objective += logp(z)

            # Generative loss
            nobj = - objective
            bits_x = nobj / (np.log(2.) * int(x.get_shape()[1]) * int(
                x.get_shape()[2]) * int(x.get_shape()[3]))  # bits per subpixel

            # Predictive loss
            if hps.weight_y > 0 and hps.ycond:

                # Classification loss
                h_y = tf.reduce_mean(z, axis=[1, 2])
                y_logits = Z.linear_zeros("classifier", h_y, hps.n_y)
                bits_y = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y_onehot, logits=y_logits) / np.log(2.)

                # Classification accuracy
                y_predicted = tf.argmax(y_logits, 1, output_type=tf.int32)
                classification_error = 1 - \
                    tf.cast(tf.equal(y_predicted, y), tf.float32)
            else:
                bits_y = tf.zeros_like(bits_x)
                classification_error = tf.ones_like(bits_x)

        return bits_x, bits_y, classification_error

    def f_loss(iterator, is_training, reuse=False):
        # f_loss: function with as input the (x,y,reuse=False), and as output a list/tuple whose first element is the loss.
        if hps.direct_iterator and iterator is not None:
            x, y = iterator.get_next()
        else:
            x, y = X, Y

        bits_x, bits_y, pred_loss = _f_loss(x, y, is_training, reuse)
        local_loss = bits_x + hps.weight_y * bits_y
        stats = [local_loss, bits_x, bits_y, pred_loss]
        global_stats = Z.allreduce_mean(
            tf.stack([tf.reduce_mean(i) for i in stats]))

        return tf.reduce_mean(local_loss), global_stats

    feeds = {'x': X, 'y': Y}
    m = abstract_model_xy(sess, hps, feeds, train_iterator,
                          test_iterator, data_init, lr, f_loss)

    # === Sampling function
    def f_sample(y, eps_std):
        with tf.variable_scope('model', reuse=True):
            y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')

            _, sample, _ = prior("prior", y_onehot, hps)
            z = sample(eps_std=eps_std)
            z = decoder(z, eps_std=eps_std)
            z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
            x = postprocess(z)

        return x

    m.eps_std = tf.placeholder(tf.float32, [None], name='eps_std')
    x_sampled = f_sample(Y, m.eps_std)

    def sample(_y, _eps_std):
        return m.sess.run(x_sampled, {Y: _y, m.eps_std: _eps_std})
    m.sample = sample


loss_train, stats_train = f_loss(train_iterator, True)

```



### Encoder and Decoder

```python
def codec(hps):
    def encoder(z, objective):
        eps = []
        for i in range(hps.n_levels):
            z, objective = revnet2d(str(i), z, objective, hps)
            if i < hps.n_levels-1:
                z, objective, _eps = split2d("pool"+str(i), z, objective=objective)
                eps.append(_eps)
        return z, objective, eps

    def decoder(z, eps=[None]*hps.n_levels, eps_std=None):
        for i in reversed(range(hps.n_levels)):
            if i < hps.n_levels-1:
                z = split2d_reverse("pool"+str(i), z, eps=eps[i], eps_std=eps_std)
            z, _ = revnet2d(str(i), z, 0, hps, reverse=True)
        return z

    return encoder, decoder
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




### `revnet2d`

```python
def revnet2d(name, z, logdet, hps, reverse=False):
    with tf.variable_scope(name):
        if not reverse:
            for i in range(hps.depth):
                z, logdet = checkpoint(z, logdet)
                z, logdet = revnet2d_step(str(i), z, logdet, hps, reverse)
            z, logdet = checkpoint(z, logdet)
        else:
            for i in reversed(range(hps.depth)):
                z, logdet = revnet2d_step(str(i), z, logdet, hps, reverse)
    return z, logdet

# Simpler, new version
@add_arg_scope
def revnet2d_step(name, z, logdet, hps, reverse):
    with tf.variable_scope(name):

        shape = Z.int_shape(z)
        n_z = shape[3]
        assert n_z % 2 == 0

        if not reverse:

            z, logdet = Z.actnorm("actnorm", z, logdet=logdet)

            if hps.flow_permutation == 0:
                z = Z.reverse_features("reverse", z)
            elif hps.flow_permutation == 1:
                z = Z.shuffle_features("shuffle", z)
            elif hps.flow_permutation == 2:
                z, logdet = invertible_1x1_conv("invconv", z, logdet)
            else:
                raise Exception()

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 += f("f1", z1, hps.width)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, hps.width, n_z)
                shift = h[:, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
                z2 += shift
                z2 *= scale
                logdet += tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

        else:

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 -= f("f1", z1, hps.width)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, hps.width, n_z)
                shift = h[:, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
                z2 /= scale
                z2 -= shift
                logdet -= tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

            if hps.flow_permutation == 0:
                z = Z.reverse_features("reverse", z, reverse=True)
            elif hps.flow_permutation == 1:
                z = Z.shuffle_features("shuffle", z, reverse=True)
            elif hps.flow_permutation == 2:
                z, logdet = invertible_1x1_conv(
                    "invconv", z, logdet, reverse=True)
            else:
                raise Exception()

            z, logdet = Z.actnorm("actnorm", z, logdet=logdet, reverse=True)

    return z, logdet

def f(name, h, width, n_out=None):
    n_out = n_out or int(h.get_shape()[3])
    with tf.variable_scope(name):
        h = tf.nn.relu(Z.conv2d("l_1", h, width))
        h = tf.nn.relu(Z.conv2d("l_2", h, width, filter_size=[1, 1]))
        h = Z.conv2d_zeros("l_last", h, n_out)
    return h

```



### Invertible 1x1 conv

```python
def invertible_1x1_conv(name, z, logdet, reverse=False):
    if True:  # Set to "False" to use the LU-decomposed version
        with tf.variable_scope(name):
            shape = Z.int_shape(z)
            w_shape = [shape[3], shape[3]]

            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                *w_shape))[0].astype('float32')

            w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)

            # dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * shape[1]*shape[2]
            dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(w, 'float64')))), 'float32') * shape[1]*shape[2]

            if not reverse:
                _w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += dlogdet
                return z, logdet
            else:
                _w = tf.matrix_inverse(w)
                _w = tf.reshape(_w, [1, 1]+w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet -= dlogdet
                return z, logdet

    else:
        # LU-decomposed version
        shape = Z.int_shape(z)
        with tf.variable_scope(name):
            dtype = 'float64'
            # Random orthogonal matrix:
            import scipy
            np_w = scipy.linalg.qr(np.random.randn(shape[3], shape[3]))[
                0].astype('float32')

            np_p, np_l, np_u = scipy.linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p = tf.get_variable("P", initializer=np_p, trainable=False)
            l = tf.get_variable("L", initializer=np_l)
            sign_s = tf.get_variable(
                "sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            # S = tf.get_variable("S", initializer=np_s)
            u = tf.get_variable("U", initializer=np_u)

            p = tf.cast(p, dtype)
            l = tf.cast(l, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u = tf.cast(u, dtype)

            w_shape = [shape[3], shape[3]]

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
            u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p, tf.matmul(l, u))

            if True:
                u_inv = tf.matrix_inverse(u)
                l_inv = tf.matrix_inverse(l)
                p_inv = tf.matrix_inverse(p)
                w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
            else:
                w_inv = tf.matrix_inverse(w)

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)

            if not reverse:
                w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += tf.reduce_sum(log_s) * (shape[1]*shape[2])
                return z, logdet
            else:
                w_inv = tf.reshape(w_inv, [1, 1]+w_shape)
                z = tf.nn.conv2d(
                    z, w_inv, [1, 1, 1, 1], 'SAME', data_format='NHWC')
                logdet -= tf.reduce_sum(log_s) * (shape[1]*shape[2])
                return z, logdet

```





## Related
- [Deep Generative Models(Part 1): Taxonomy and VAEs](https://arxivnote.ddlee.cn/Deep-Generative-Models-Taxonomy-VAE.html)
- [Deep Generative Models(Part 2): Flow-based Models(include PixelCNN)](https://arxivnote.ddlee.cn/Deep-Generative-Models-Flow-based-Models-PixelCNN.html)
- [Deep Generative Models(Part 3): GANs](https://arxivnote.ddlee.cn/Deep-Generative-Models-GAN-WGAN-SAGAN-StyleGAN-BigGAN.html)
- Gated PixelCNN: [Conditional Image Generation with PixelCNN Decoders - van den Oord - NIPS 2016](https://arxivnote.ddlee.cn/Conditional-Image-Generation-with-PixelCNN-Decoders.html)
- PixelRNN & PixelCNN: [Pixel Recurrent Neural Networks - van den Oord - ICML 2016](https://arxivnote.ddlee.cn/Pixel-Recurrent-Neural-Networks.html)
- VQ-VAE: [Neural Discrete Representation Learning - van den Oord - NIPS 2017](https://arxivnote.ddlee.cn/Neural-Discrete-Representation-Learning.html)
- VQ-VAE-2: [ Generating Diverse High-Fidelity Images with VQ-VAE-2 - Razavi - 2019](https://arxivnote.ddlee.cn/Generating-Diverse-High-Fidelity-Images-with-VQ-VAE-2.html)