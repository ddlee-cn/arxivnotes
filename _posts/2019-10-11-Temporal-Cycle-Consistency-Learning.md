---
title: Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019
tag:
- Representation Learning
- Self-Supervised
- Video
---

## Info

- Title: Temporal Cycle-Consistency Learning
- Task: Representation Learning
- Author: Debidatta Dwibedi, Yusuf Aytar , Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman
- Date: Apr. 2019
- Arxiv: [1904.07846](https://arxiv.org/abs/1904.07846)
- Published: CVPR 2019



## Highlights

Temporal cycle-consistency (TCC) loss for video alignment in learned latent space with a self-supervised manner.

## Abstract

We introduce a self-supervised representation learning method based on the task of temporal alignment between videos. The method trains a network using temporal cycle consistency (TCC), a differentiable cycle-consistency loss that can be used to find correspondences across time in multiple videos. The resulting per-frame embeddings can be used to align videos by simply matching frames using the nearest-neighbors in the learned embedding space.

To evaluate the power of the embeddings, we densely label the Pouring and Penn Action video datasets for action phases. We show that (i) the learned embeddings enable few-shot classification of these action phases, significantly reducing the supervised training requirements; and (ii) TCC is complementary to other methods of self-supervised learning in videos, such as Shuffle and Learn and Time-Contrastive Networks. The embeddings are also used for a number of applications based on alignment (dense temporal correspondence) between video pairs, including transfer of metadata of synchronized modalities between videos (sounds, temporal semantic labels), synchronized playback of multiple videos, and anomaly detection.



## Motivation & Design





![Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://i.imgur.com/NGSuQcC.png)



The authors present a self-supervised representation learning technique called temporal cycle consistency (TCC) learning. It is inspired by the temporal video alignment problem, which refers to the task of finding correspondences across multiple videos despite many factors of variation. The learned representations are useful for fine-grained temporal understanding in videos. Additionally, we can now align multiple videos by simply finding nearest-neighbor frames in the embedding space.



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


### Cycle-consistent representation learning

![Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://i.imgur.com/P5a3w2W.png)



Ther are two example video sequences encoded in an example embedding space. If we use nearest neighbors for matching, one point (shown in black) is cycling back to itself while another one (shown in red) is not. Our target is to learn an embedding space where maximum number of points can cycle back to themselves. We achieve it by minimizing the cycle consistency error (shown in red dotted line) for each point in every pair of sequences.



![Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://i.imgur.com/Ct2zbuA.png)



The embedding sequences $U$ and $V$are obtained by encoding video sequences S and $T$ with the encoder network $φ$, respectively. For the selected point $u_i$ in $U$ , soft nearest neighbor computation and cycling back to $U$ again is demonstrated visually. Finally the normalized distance between the index $i$ and cycling back distribution $N (μ, σ2)$ (which is fitted to $β$) is minimized.




## Experiments & Ablation Study


### Phase Classification

![Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://i.imgur.com/PXThdLI.png)


### t-SNE Visualization of Leanred Embeddings

![Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://i.imgur.com/7wUm5ly.png)

## Code

[Project Site](https://sites.google.com/view/temporal-cycle-consistency)

[TensorFlow](https://github.com/google-research/google-research/tree/master/tcc)


### Generating Cycle Pairs and Computing Cycle-Consistent Loss
```python
def gen_cycles(num_cycles, batch_size, cycle_length=2):
  """Generates cycles for alignment.
  Generates a batch of indices to cycle over. For example setting num_cycles=2,
  batch_size=5, cycle_length=3 might return something like this:
  cycles = [[0, 3, 4, 0], [1, 2, 0, 3]]. This means we have 2 cycles for which
  the loss will be calculated. The first cycle starts at sequence 0 of the
  batch, then we find a matching step in sequence 3 of that batch, then we
  find matching step in sequence 4 and finally come back to sequence 0,
  completing a cycle.
  """
  sorted_idxes = tf.tile(tf.expand_dims(tf.range(batch_size), 0),
                         [num_cycles, 1])
  sorted_idxes = tf.reshape(sorted_idxes, [batch_size, num_cycles])
  cycles = tf.reshape(tf.random.shuffle(sorted_idxes),
                      [num_cycles, batch_size])
  cycles = cycles[:, :cycle_length]
  # Append the first index at the end to create cycle.
  cycles = tf.concat([cycles, cycles[:, 0:1]], axis=1)
  return cycles


def compute_stochastic_alignment_loss(embs,steps,seq_lens,num_steps,batch_size,loss_type,similarity_type,num_cycles,cycle_length,temperature,label_smoothing,variance_lambda,huber_delta,normalize_indices):
  """Compute cycle-consistency loss by stochastically sampling cycles.
  """
  # Generate cycles.
  cycles = gen_cycles(num_cycles, batch_size, cycle_length)

  logits, labels = _align(cycles, embs, num_steps, num_cycles, cycle_length,
                          similarity_type, temperature)

  if loss_type == 'classification':
    loss = classification_loss(logits, labels, label_smoothing)
  elif 'regression' in loss_type:
    steps = tf.gather(steps, cycles[:, 0])
    seq_lens = tf.gather(seq_lens, cycles[:, 0])
    loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
                           loss_type, normalize_indices, variance_lambda,
                           huber_delta)
  else:
    raise ValueError('Unidentified loss type %s. Currently supported loss '
                     'types are: regression_mse, regression_huber, '
                     'classification .'
                     % loss_type)
  return loss
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





### Find Nearest Neighbor and Align

```python
def _align_single_cycle(cycle, embs, cycle_length, num_steps,
                        similarity_type, temperature):
  """Takes a single cycle and returns logits (simialrity scores) and labels."""
  # Choose random frame.
  n_idx = tf.random_uniform((), minval=0, maxval=num_steps, dtype=tf.int32)
  # Create labels
  onehot_labels = tf.one_hot(n_idx, num_steps)

  # Choose query feats for first frame.
  query_feats = embs[cycle[0], n_idx:n_idx+1]

  num_channels = tf.shape(query_feats)[-1]
  for c in range(1, cycle_length+1):
    candidate_feats = embs[cycle[c]]

    if similarity_type == 'l2':
      # Find L2 distance.
      mean_squared_distance = tf.reduce_sum(
          tf.squared_difference(tf.tile(query_feats, [num_steps, 1]),
                                candidate_feats), axis=1)
      # Convert L2 distance to similarity.
      similarity = -mean_squared_distance

    elif similarity_type == 'cosine':
      # Dot product of embeddings.
      similarity = tf.squeeze(tf.matmul(candidate_feats, query_feats,
                                        transpose_b=True))
    else:
      raise ValueError('similarity_type can either be l2 or cosine.')

    # Scale the distance  by number of channels. This normalization helps with
    # optimization.
    similarity /= tf.cast(num_channels, tf.float32)
    # Scale the distance by a temperature that helps with how soft/hard the
    # alignment should be.
    similarity /= temperature

    beta = tf.nn.softmax(similarity)
    beta = tf.expand_dims(beta, axis=1)
    beta = tf.tile(beta, [1, num_channels])

    # Find weighted nearest neighbour.
    query_feats = tf.reduce_sum(beta * candidate_feats,
                                axis=0, keepdims=True)

  return similarity, onehot_labels


def _align(cycles, embs, num_steps, num_cycles, cycle_length,
           similarity_type, temperature):
  """Align by finding cycles in embs."""
  logits_list = []
  labels_list = []
  for i in range(num_cycles):
    logits, labels = _align_single_cycle(cycles[i],
                                         embs,
                                         cycle_length,
                                         num_steps,
                                         similarity_type,
                                         temperature)
    logits_list.append(logits)
    labels_list.append(labels)

  logits = tf.stack(logits_list)
  labels = tf.stack(labels_list)

  return logits, labels

```



## Related

- [Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://arxivnote.ddlee.cn/2019/09/30/Test-Time-Training-Generalization.html)
- [Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://arxivnote.ddlee.cn/2019/10/11/Temporal-Cycle-Consistency-Learning.html)
- [Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://arxivnote.ddlee.cn/2019/09/30/Test-Time-Training-Generalization.html)
- [Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://arxivnote.ddlee.cn/2019/09/29/Learning-Correspondence-Cycle-Consistency-Time.html)

