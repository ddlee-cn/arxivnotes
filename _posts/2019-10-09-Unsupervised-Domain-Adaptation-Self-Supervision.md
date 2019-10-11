---
title: Unsupervised Domain Adaptation through Self-Supervision - Yu Sun - 2019
tag:
- Self-Supervised
- Domain Adaptation

---

## Info

- Title: Unsupervised Domain Adaptation through Self-Supervision
- Task: Domain Adaptaion
- Author: Yu Sun, Eric Tzeng, Trevor Darrell, Alexei A. Efros
- Date: Sep. 2019
- Arxiv: [1909.11825](https://arxiv.org/abs/1909.11825)

  

## Abstract

This paper addresses unsupervised domain adaptation, the setting where labeled training data is available on a source domain, but the goal is to have good performance on a target domain with only unlabeled data. Like much of previous work, we seek to align the learned representations of the source and target domains while preserving discriminability. The way we accomplish alignment is by learning to perform auxiliary self-supervised task(s) on both domains simultaneously. Each self-supervised task brings the two domains closer together along the direction relevant to that task. Training this jointly with the main task classifier on the source domain is shown to successfully generalize to the unlabeled target domain. The presented objective is straightforward to implement and easy to optimize. We achieve state-of-the-art results on four out of seven standard benchmarks, and competitive results on segmentation adaptation. We also demonstrate that our method composes well with another popular pixel-level adaptation method.



## Motivation & Design

The proposed method jointly trains a supervised head on labeled source data and self-supervised
heads on unbaled data from both domains. The heads use high-level features from a shared encoder,
which learns to align the feature distributions.



![Unsupervised Domain Adaptation through Self-Supervision - Yu Sun - 2019](https://i.imgur.com/oa62GMx.png)



Each loss function corresponds to a different “head” $hk$ for $k = 0...K$, which produces predictions
in the respective label space. All the task-specific heads (including $h_0$ for the actual prediction task)
share a common feature extractor $φ$. Altogether, the parameters of $φ$ and $h_k$ , $k = 0, ..k$ are the
learned variables i.e. the free parameters of our optimization problem.

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


## Experiments & Ablation Study

### Results on MNIST→MNIST-M (left) and CIFAR-10→STL-10 (right)

![Unsupervised Domain Adaptation through Self-Supervision - Yu Sun - 2019](https://i.imgur.com/EQBDXLm.png)

Test error converges smoothly on the source and target domains for the main task as well as the self-supervised task. This kind of smooth convergence is often seen in supervised learning, but rarely in adversarial learning. The centroid distance (linear MMD) between the feature distributions of the two domains converges alongside, even though it is never explicitly optimized for.

### Test accuracy (%) on standard domain adaptation benchmarks

![Unsupervised Domain Adaptation through Self-Supervision - Yu Sun - 2019](https://i.imgur.com/sx4huGA.png)

The results are organized according to the self-supervised task(s) used: R for rotation, L for location, and F for flip. The authors achieve state-of-the-art accuracy on four out of the seven benchmarks.





## Code

[PyTorch](https://github.com/yueatsprograms/uda_release)



### Training Process

```python
net.train()
for sstask in sstasks:
    sstask.head.train()
    sstask.scheduler.step()

epoch_stats = []
for batch_idx, (sc_tr_inputs, sc_tr_labels) in enumerate(sc_tr_loader):
    for sstask in sstasks:
        sstask.train_batch()

    sc_tr_inputs, sc_tr_labels = sc_tr_inputs.cuda(), sc_tr_labels.cuda()
    optimizer_cls.zero_grad()
    outputs_cls = net(sc_tr_inputs)
    loss_cls = criterion_cls(outputs_cls, sc_tr_labels)
    loss_cls.backward()
    optimizer_cls.step()
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




### Self-Supervised Tasks

```python
class SSTask():
	def __init__(self, ext, head, criterion, optimizer, scheduler,
				 su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader):
		pass # omitted

	def train_batch(self):
		su_tr_inputs, su_tr_labels = next(self.su_tr_loader_iterator)
		tu_tr_inputs, tu_tr_labels = next(self.tu_tr_loader_iterator)
		self.su_tr_iter_counter += 1
		self.tu_tr_iter_counter += 1

		us_tr_inputs = torch.cat((su_tr_inputs, tu_tr_inputs))
		us_tr_labels = torch.cat((su_tr_labels, tu_tr_labels))
		us_tr_inputs, us_tr_labels = us_tr_inputs.cuda(), us_tr_labels.cuda()

		self.optimizer.zero_grad()
		outputs = self.ext(us_tr_inputs)
		outputs = self.head(outputs)
		loss = self.criterion(outputs, us_tr_labels)
		loss.backward()
		self.optimizer.step()
		
		if self.su_tr_iter_counter > len(self.su_tr_loader):
			self.su_tr_epoch_counter += 1
			self.reset_su()
		if self.tu_tr_iter_counter > len(self.tu_tr_loader):
			self.tu_tr_epoch_counter += 1
			self.reset_tu()
            
# Usage
net = ResNet(args.depth, args.width, classes=classes, channels=channels).cuda()
ext = extractor_from_layer3(net)

su_tr_dataset = DsetSSRotRand(DsetNoLabel(sc_tr_dataset), digit=digit)
su_te_dataset = DsetSSRotRand(DsetNoLabel(sc_te_dataset), digit=digit)
su_tr_loader = torchdata.DataLoader(su_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
su_te_loader = torchdata.DataLoader(su_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

tu_tr_dataset = DsetSSRotRand(DsetNoLabel(tg_tr_dataset), digit=digit)
tu_te_dataset = DsetSSRotRand(DsetNoLabel(tg_te_dataset), digit=digit)
tu_tr_loader = torchdata.DataLoader(tu_tr_dataset, batch_size=args.batch_size//2, shuffle=True, num_workers=4)
tu_te_loader = torchdata.DataLoader(tu_te_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=4)

head = linear_on_layer3(4, args.width, 8).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(list(ext.parameters()) + list(head.parameters()), 
                        lr=args.lr_rotation, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, [args.milestone_1, args.milestone_2], gamma=0.1, last_epoch=-1)
sstask = SSTask(ext, head, criterion, optimizer, scheduler,
             su_tr_loader, su_te_loader, tu_tr_loader, tu_te_loader)
```



### Self-Supervised Task Head

```python
class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

def extractor_from_layer3(net):
	layers = [net.conv1, net.layer1, net.layer2, net.layer3, net.bn, net.relu]
	return nn.Sequential(*layers)

def linear_on_layer3(classes, width, pool):
	layers = [nn.AvgPool2d(pool), ViewFlatten(), nn.Linear(64 * width, classes)]
	return nn.Sequential(*layers)
```





## Related

- [Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://arxivnote.ddlee.cn/2019/09/30/Test-Time-Training-Generalization.html)
- [Temporal Cycle-Consistency Learning - Dwibedi - CVPR 2019](https://arxivnote.ddlee.cn/2019/10/11/Temporal-Cycle-Consistency-Learning.html)
- [Test-Time Training for Out-of-Distribution Generalization - Yu Sun - 2019](https://arxivnote.ddlee.cn/2019/09/30/Test-Time-Training-Generalization.html)
- [Learning Correspondence from the Cycle-Consistency of Time - Xiaolong Wang - CVPR 2019](https://arxivnote.ddlee.cn/2019/09/29/Learning-Correspondence-Cycle-Consistency-Time.html)

