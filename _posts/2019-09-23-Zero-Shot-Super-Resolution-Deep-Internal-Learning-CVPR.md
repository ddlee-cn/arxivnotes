---
title: Zero-Shot Super-Resolution using Deep Internal Learning
tag:
- Super-Resolution
- Zero-Shot
- Low-Level Vision
---

## Info

- Title: "Zero-Shot" Super-Resolution using Deep Internal Learning
- Task: Super-Resolution
- Author: Assaf Shocher, Nadav Cohen, Michal Irani
- Date: Dec. 2017
- Arxiv: [1712.06087](http://arxiv.org/abs/1712.06087)
- Published: CVPR 2018



## Highlights

- Test-time training for image-specific CNN.

- It can handle non-ideal imaging conditions, and a wide variety of images and data types (even if encountered for the first time).

- It does not require pretraining and can be run with modest amounts of computational resources.

- It can be applied for SR to any size and theoretically also with any aspect-ratio.

  

## Abstract

Deep Learning has led to a dramatic leap in Super-Resolution (SR) performance in the past few years. However, being supervised, these SR methods are restricted to specific training data, where the acquisition of the low-resolution (LR) images from their high-resolution (HR) counterparts is predetermined (e.g., bicubic downscaling), without any distracting artifacts (e.g., sensor noise, image compression, non-ideal PSF, etc). Real LR images, however, rarely obey these restrictions, resulting in poor SR results by SotA (State of the Art) methods. In this paper we introduce "Zero-Shot" SR, which exploits the power of Deep Learning, but does not rely on prior training. We exploit the internal recurrence of information inside a single image, and train a small image-specific CNN at test time, on examples extracted solely from the input image itself. As such, it can adapt itself to different settings per image. This allows to perform SR of real old photos, noisy images, biological data, and other images where the acquisition process is unknown or non-ideal. On such images, our method outperforms SotA CNN-based SR methods, as well as previous unsupervised SR methods. To the best of our knowledge, this is the first unsupervised CNN-based SR method.



## Motivation & Design

Fundamental to our approach is the fact that natural images have strong internal data repetition. For example, small image patches (e.g., 5×5, 7×7) were shown to repeat many times inside a single image, both within the same scale, as well as across different image scales. 

In fact, the only evidence to the existence of these tiny handrails exists internally, inside this image, at a different location and different scale. It cannot be found in any external database of examples, no matter how large this dataset is!

![ZSSR](https://github.com/assafshocher/ZSSR/raw/master/figs/sketch.png)


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



Our image-specific CNN leverages on the power of the cross-scale internal recurrence of image-specific information, without being restricted by the above-mentioned limitations of patch-based methods. We train a CNN to infer complex image-specific HR-LR relations from the LR image and its downscaled versions (self-supervision). We then apply those learned relations on the LR input image to produce the HR output. This outperforms unsupervised patchbased SR by a large margin.



![ZSSR](<https://i.imgur.com/awZKqQ4.png>)

(a) Externally-supervised SR CNNs are pre-trained on large external databases of images. The resulting very deep network is then applied to the test image I. 

(b) Our proposed method (ZSSR): a small image-specific CNN is trained on examples extracted internally, from the test image itself. It learns how to recover the test image I from its coarser resolutions. The resulting self-supervised network is then applied to the LR image I to produce its HR output.



## Experiments & Ablation Study

Our network can be adapted to the specific degradations/settings of the test image at hand, at test time. Our network can receive from the user, at test time, any of the following parameters:
(i) The desired downscaling kernel (when no kernel is provided, the bicubic kernel serves as a default).
(ii) The desired SR scale-factors.
(iii) The desired number of gradual scale increases (a trade-off between speed and quality – the default is 6).
(iv) Whether to enforce Backprojection between the LR and HR image (the default is ‘Yes’).
(v) Whether to add ‘noise’ to the LR sons in each LR-HR example pair extracted from the LR test image (the default is ‘No’).



![ZSSR Performance](<https://i.imgur.com/IvNOkX2.png>)



More detailed comparison with supervised methods can be found at [Project Site](<http://www.wisdom.weizmann.ac.il/~vision/zssr/>).



## Code

[Project Site](<http://www.wisdom.weizmann.ac.il/~vision/zssr/>)

[TensorFlow](<https://github.com/assafshocher/ZSSR>)



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


### Core Design

#### Data Preparation

```python
# Use augmentation from original input image to create current father.
# If other scale factors were applied before, their result is also used (hr_fathers_in)
self.hr_father = random_augment(ims=self.hr_fathers_sources,
                                base_scales=[1.0] + self.conf.scale_factors,
                                leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                                no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                                min_scale=self.conf.augment_min_scale,
                                max_scale=([1.0] + self.conf.scale_factors)[len(self.hr_fathers_sources)-1],
                                allow_rotation=self.conf.augment_allow_rotation,
                                scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                                shear_sigma=self.conf.augment_shear_sigma,
                                crop_size=self.conf.crop_size)

# Get lr-son from hr-father
self.lr_son = self.father_to_son(self.hr_father)

def father_to_son(self, hr_father):
    # Create son out of the father by downscaling and if indicated adding noise
    lr_son = imresize(hr_father, 1.0 / self.sf, kernel=self.kernel)
    return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)
```



### Network

```python
def build_network(self, meta):
    with self.model.as_default():

        # Learning rate tensor
        self.learning_rate_t = tf.placeholder(tf.float32, name='learning_rate')

        # Input image
        self.lr_son_t = tf.placeholder(tf.float32, name='lr_son')

        # Ground truth (supervision)
        self.hr_father_t = tf.placeholder(tf.float32, name='hr_father')

        # Filters
        self.filters_t = [tf.get_variable(shape=meta.filter_shape[ind], name='filter_%d' % ind,
                                          initializer=tf.random_normal_initializer(
                                              stddev=np.sqrt(meta.init_variance/np.prod(
                                                  meta.filter_shape[ind][0:3]))))
                          for ind in range(meta.depth)]

        # Activate filters on layers one by one (this is just building the graph, no calculation is done here)
        self.layers_t = [self.lr_son_t] + [None] * meta.depth
        for l in range(meta.depth - 1):
            self.layers_t[l + 1] = tf.nn.relu(tf.nn.conv2d(self.layers_t[l], self.filters_t[l],
                                                           [1, 1, 1, 1], "SAME", name='layer_%d' % (l + 1)))

        # Last conv layer (Separate because no ReLU here)
        l = meta.depth - 1
        self.layers_t[-1] = tf.nn.conv2d(self.layers_t[l], self.filters_t[l],
                                         [1, 1, 1, 1], "SAME", name='layer_%d' % (l + 1))

        # Output image (Add last conv layer result to input, residual learning with global skip connection)
        self.net_output_t = self.layers_t[-1] + self.conf.learn_residual * self.lr_son_t

        # Final loss (L1 loss between label and output layer)
        self.loss_t = tf.reduce_mean(tf.reshape(tf.abs(self.net_output_t - self.hr_father_t), [-1]))

        # Apply adam optimizer
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_t).minimize(self.loss_t)
        self.init_op = tf.initialize_all_variables()
```



#### Training Process

```python
def forward_backward_pass(self, lr_son, hr_father):
    # First gate for the lr-son into the network is interpolation to the size of the father
    # Note: we specify both output_size and scale_factor. best explained by example: say father size is 9 and sf=2,
    # small_son size is 4. if we upscale by sf=2 we get wrong size, if we upscale to size 9 we get wrong sf.
    # The current imresize implementation supports specifying both.
    interpolated_lr_son = imresize(lr_son, self.sf, hr_father.shape, self.conf.upscale_method)

    # Create feed dict
    feed_dict = {'learning_rate:0': self.learning_rate,
                 'lr_son:0': np.expand_dims(interpolated_lr_son, 0),
                 'hr_father:0': np.expand_dims(hr_father, 0)}

    # Run network
    _, self.loss[self.iter], train_output = self.sess.run([self.train_op, self.loss_t, self.net_output_t],
                                                          feed_dict)
    return np.clip(np.squeeze(train_output), 0, 1)
```





## Related

- [Deep Image Prior - Ulyanov - CVPR 2018](https://arxivnote.ddlee.cn/2019/08/26/Deep-Image-Prior-Ulyanov-CVPR-2018.html)
- [The Perception-Distortion Tradeoff - Blau - CVPR 2018 - Matlab](https://arxivnote.ddlee.cn/2019/08/12/The-Perception-Distortion-Tradeoff-Blau-CVPR.html)
- [Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff - Blau - ICML 2019](https://arxivnote.ddlee.cn/2019/08/15/Rethinking-Lossy-Compression-The-Rate-Distortion-Perception-Tradeoff-Blau-ICML.html)