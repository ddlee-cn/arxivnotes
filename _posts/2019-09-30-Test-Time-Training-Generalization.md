---
title: Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019
tag:
- Theoretical
- Domain Adaptation

---

## Info

- Title: Test-Time Training for Out-of-Distribution Generalization

- Author: Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, Moritz Hardt

- Date: Sep. 2019

- Arxiv: [1909.13231](https://arxiv.org/abs/1909.13231)

  

## Highlights

- Theoretically prove of benefits on test-time self-supervised training for convex loss functions, empirically for non-convex functions.



## Abstract

We introduce a general approach, called test-time training, for improving the performance of predictive models when test and training data come from different distributions. Test-time training turns a single unlabeled test instance into a self-supervised learning problem, on which we update the model parameters before making a prediction on the test sample. We show that this simple idea leads to surprising improvements on diverse image classification benchmarks aimed at evaluating robustness to distribution shifts. Theoretical investigations on a convex model reveal helpful intuitions for when we can expect our approach to help.



## Motivation & Design

### Method Overview

The route we take is to create a self-supervised learning problem based only on this single test case x, which we use to update θ at test-time before we then make a prediciton on x. Self-supervised learning uses an auxiliary task that automatically creates labels from unlabeled data. For the visual data we work with, the task rotates an image x by a multiple of 90 degrees, and assigns the angle as the label.



### Theorem(for convex function)

![Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://i.imgur.com/XVfZgOM.png)





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



### Empirical Results(for non-convex function)

#### The Corruption Benchmark

Hendrycks & Dietterich (2019) propose to benchmark robustness of neural networks on 15 types of corruptions from four broad categories: noise, blur, weather and digital. Each corruption type comes in five levels of severity, with level 5 the most severe. The corruptions are algorithmically simulated to mimic real-world corruptions as much as possible on copies of the test set for both CIFAR-10 and ImageNet.

![Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://i.imgur.com/Oue9osr.png)



Scatter plot of the inner product between the gradients (on the shared feature extractor θe ) of the main task $l_m$ and the self-supervised task $l_e$ , and the improvement in test error (%) from test-time training, for the standard (left) and online (right) version. Each point is the average over a test set, and each scatter plot has 75 test sets, from all 15 types of corruptions over five levels. The blue lines and bands are the best linear fits and the 99% confidence intervals. The linear correlation coefficients are 0.93 and 0.89 respectively, indicating strong positive correlation between the two quantities, as suggested by Theorem 1.



## Experiments & Ablation Study



### Test Error on CIFAR-10-C, level 5

![Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://i.imgur.com/NMC1HWR.png)



### Test Accuracy on ImageNet-C, level 5

![Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://i.imgur.com/haxblUF.png)

The lower panel shows the accuracy of the online version as the average over a sliding window of 100 samples; test-time learning online generalizes better as more samples are tested on, without hurting on the original distribution. We use accuracy instead of error here because the baseline performance is very poor with most corruptions.



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

[Cifar](https://github.com/yueatsprograms/ttt_cifar_release)

[ImageNet](https://github.com/yueatsprograms/ttt_cifar_release)



### Test-Time Training with Rotation as Auxiliary Task

add an additional step after main task prediction

```python
for batch_idx, (inputs, labels) in enumerate(trloader):
    optimizer.zero_grad()
    inputs_cls, labels_cls = inputs.cuda(), labels.cuda()
    outputs_cls = net(inputs_cls)
    loss = criterion(outputs_cls, labels_cls)

    inputs_ssh, labels_ssh = rotate_batch(inputs, args.rotation_type)
    outputs_ssh = ssh(inputs_ssh)
    loss_ssh = criterion(outputs_ssh, labels_ssh)
    loss += loss_ssh

    loss.backward()
    optimizer.step()
```



### Rotation Label Generation

```python
def rotate_batch(batch, label):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	elif label == 'expand':
		labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
					torch.zeros(len(batch), dtype=torch.long) + 1,
					torch.zeros(len(batch), dtype=torch.long) + 2,
					torch.zeros(len(batch), dtype=torch.long) + 3])
		batch = batch.repeat((4,1,1,1))
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels
```



### Self-Supervised Head(SSH) Model Architecture

Extracting head from backbone

```python
net = ResNet(args.depth, args.width, channels=3, classes=classes, norm_layer=norm_layer).cuda()
ext = extractor_from_layer2(net)
head = head_on_layer2(net, args.width, 4)
ssh = ExtractorHead(ext, head)

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		return self.head(self.ext(x))

def extractor_from_layer2(net):
	layers = [net.conv1, net.layer1, net.layer2]
	return nn.Sequential(*layers)

def head_on_layer2(net, width, classes):
	head = copy.deepcopy([net.layer3, net.bn, net.relu, net.avgpool])
	head.append(ViewFlatten())
	head.append(nn.Linear(64 * width, classes))
	return nn.Sequential(*head)
```





## Related

- [Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://arxivnote.ddlee.cn/2019/09/30/Test-Time-Training-Generalization.html)
- [Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://arxivnote.ddlee.cn/2019/10/11/Temporal-Cycle-Consistency-Learning.html)
- [Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://arxivnote.ddlee.cn/2019/09/30/Test-Time-Training-Generalization.html)
- [Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://arxivnote.ddlee.cn/2019/09/29/Learning-Correspondence-Cycle-Consistency-Time.html)

