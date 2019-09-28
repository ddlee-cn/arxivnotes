---
title: "Image Inpainting: From PatchMatch to Pluralistic"
tag:
- Image Inpainting
- Review
---



## Traditional

### PatchMatch A Randomized Correspondence Algorithm for Structural Image Editing - SIGGRAPH 2009

![CleanShot 2019-09-22 at 14.59.37@2x](https://i.imgur.com/qfAuKx8.jpg)


This paper presents interactive image editing tools using a new randomized algorithm for quickly finding approximate nearest-neighbor matches between image patches. Previous research in graphics and vision has leveraged such nearest-neighbor searches to provide a variety of high-level digital image editing tools. However, the cost of computing a field of such matches for an entire image has eluded previous efforts to provide interactive performance. Our algorithm offers substantial performance improvements over the previous state of the art (20-100x), enabling its use in interactive editing tools. The key insights driving the algorithm are that some good patch matches can be found via random sampling, and that natural coherence in the imagery allows us to propagate such matches quickly to surrounding areas. We offer theoretical analysis of the convergence properties of the algorithm, as well as empirical and practical evidence for its high quality and performance. This one simple algorithm forms the basis for a variety of tools -- image retargeting, completion and reshuffling -- that can be used together in the context of a high-level image editing application. Finally, we propose additional intuitive constraints on the synthesis process that offer the user a level of control unavailable in previous methods.

## Learning-based

### Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017

![Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://i.imgur.com/8vc75lO.jpg)

**Overview of network architecture**
It consists of a completion network and two auxiliary context discriminator networks that are used only for training the completion network and are not used during the testing. The global discriminator network takes the entire image as input, while the local discriminator network takes only a small region around the completed area as input. Both discriminator networks are trained to determine if an image is real or completed by the completion network, while the completion network is trained to fool both discriminator networks.

![Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://i.imgur.com/RbL82jP.jpg)

### Generative Image Inpainting with Contextual Attention - Yu - CVPR 2018
Overview of our improved generative inpainting framework. The coarse network is trained with reconstruction loss explicitly, while the refinement network is trained with reconstruction loss, global and local WGAN-GP adversarial loss.

![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/Tw2ryjB.jpg)

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


Illustration of the contextual attention layer. Firstly we use convolution to compute matching score of foreground patches with background patches (as convolu- tional filters). Then we apply softmax to compare and get attention score for each pixel. Finally we reconstruct fore- ground patches with background patches by performing de- convolution on attention score. The contextual attention layer is differentiable and fully-convolutional.

![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/PvfMyPW.jpg)

Based on coarse result from the first encoder- decoder network, two parallel encoders are introduced and then merged to single decoder to get inpainting result. For visualization of attention map, color indicates relative loca- tion of the most interested background patch for each pixel in foreground. For examples, white (center of color coding map) means the pixel attends on itself, pink on bottom-left, green means on top-right.

![Generative Image Inpainting with Contextual Attention](https://i.imgur.com/WOFEv0p.jpg)

### Pluralistic Image Completion - CVPR 2019
![CleanShot 2019-09-22 at 14.53.16@2x](https://i.imgur.com/q7l9lY5.jpg)


Overview of our architecture with two parallel pipelines. The reconstructive pipeline (yellow line) combines information from Im and Ic, which is used only for training. The generative pipeline (blue line) infers the conditional distribution of hidden regions, that can be sampled during testing. Both representation and generation networks share identical weights.


### EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning - Nazeri - ICCV 2019 
The spirit: “lines first, color next”, which is partly in- spired by our understanding of how artists work.

![EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://i.imgur.com/CHtT5JM.jpg)

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


## Appendix

A reference paper list for image inpainting(from [geekyutao/Image-Inpainting](https://github.com/geekyutao/Image-Inpainting))


### Traditional Methods

Year|Proceeding|Title|Comment
----|----|------|----------
2000|SIGGRAPH 2000|**Image Inpainting**  [[pdf]](http://slipguru.disi.unige.it/readinggroup/papers_vis/bertalmio00inpainting.pdf)  |Diffusion-based
2001|TIP 2001|Filling-in by joint interpolation of vector fields and gray levels [[pdf]](https://conservancy.umn.edu/bitstream/handle/11299/3462/1/1706.pdf)|Diffusion-based
2001|CVPR 2001|Navier-stokes, ﬂuid dynamics, and image and video inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=990497)|
2001|SIGGRAPH 2001|Image Quilting for Texture Synthesis and Transfer [[pdf]](https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf)  |
2001|SIGGRAPH 2001|Synthesizing Natural Textures [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.359.8241&rep=rep1&type=pdf)|
2002|EJAM 2002|Digital inpainting based on the mumford–shah–euler image model [[pdf]](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/26ACC4694C7F064B6F40D55C09ACA9A1/S0956792502004904a.pdf/digital_inpainting_based_on_the_mumfordshaheuler_image_model.pdf)  |Diffusion-based
2003|CVPR 2003| Object removal by exemplar-based inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1211538)|
2003|TIP 2003|Simultaneous structure and texture image inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1217265)|Diffusion-based
2003|TIP 2003|Structure and Texture Filling-In of Missing Image Blocks in Wireless Transmission and Compression Applications [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1197835)|
2003|ICCV 2003|Learning How to Inpaint from Global Image Statistics [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1238360)|Diffusion-based
2003|TOG 2003|Fragment-based image completion [[pdf]](http://delivery.acm.org/10.1145/890000/882267/p303-drori.pdf?ip=222.195.92.10&id=882267&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EA4F9C023AC60E700%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1553430113_8d3cc7f5adde2fb3894043de791d9150) | Patch-based
2004|TIP 2004|**Region Filling and Object Removal by Exemplar-Based Image Inpainting** [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/criminisi_tip2004.pdf)|Patch-based; Inpainting order
2004|TPAMI 2004|Space-Time Video Completion [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1315022) |
2005|SIGGRAPH 2005|Image Completion with Structure Propagation [[pdf]](http://jiansun.org/papers/ImageCompletion_SIGGRAPH05.pdf)|Patch-based
2006|ISCS 2006|Image Compression with Structure Aware Inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1692960)|
2007|TOG 2007| Scene completion using millions of photographs [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.299.518&rep=rep1&type=pdf)|   
2007|CSVT 2007|Image Compression With Edge-Based Inpainting [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/inpainting_csvt_07.pdf)|Diffusion-based
2008|CVPR 2008|Summarizing Visual Data Using Bidirectional Similarity [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4587842)|
2009|SIGGRAPH 2009|**PatchMatch: a randomized correspondence algorithm for structural image editing** [[pdf]](http://www.faculty.idc.ac.il/arik/seminar2009/papers/patchMatch.pdf)  |Patch-based
2010| TIP 2010|Image inpainting by patch propagation using patch sparsity [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5404308) |Patch-based
2011|FTCGV 2011|Structured learning and prediction in computer vision [[pdf]](http://pub.ist.ac.at/~chl/papers/nowozin-fnt2011.pdf)|
2011|ICIP 2011|Examplar-based inpainting based on local geometry [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6116441)|Inpainting order
2012|TOG 2012|Combining inconsistent images using patch-based synthesis[[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.364.5147&rep=rep1&type=pdf)|Patch-based
2014|TOG 2014|**Image completion using Planar structure guidance** [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/structure_completion_small.pdf)|Patch-based
2014|TVCG 2014|High-Quality Real-Time Video Inpainting with PixMix [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6714519)|Video
2014|SIAM 2014|Video inpainting of complex scenes [[pdf]](https://arxiv.org/pdf/1503.05528.pdf)|Video
2015|TIP 2015|Annihilating Filter-Based Low-Rank Hankel Matrix Approach for Image Inpainting [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7127011)|
2015|TIP 2015|**Exemplar-Based Inpainting: Technical Review and New Heuristics for Better Geometric Reconstructions** [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7056453)|
2016|TOG 2016|Temporally coherent completion of dynamic video [[pdf]](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/SigAsia_2016_VideoCompletion.pdf) | Video

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


### Deep-Learning-based Method



Year|Proceeding|Title|Comment
------|-----|-----|---------
2012|NIPS 2012| Image denoising and inpainting with deep neural networks [[pdf]](http://papers.nips.cc/paper/4686-image-denoising-and-inpainting-with-deep-neural-networks.pdf)|
2014|GCPR 2014|Mask-specific inpainting with deep neural networks [[pdf]](https://link.springer.com/chapter/10.1007/978-3-319-11752-2_43)|
2014|NIPS 2014|Deep Convolutional Neural Network for Image Deconvolution [[pdf]](http://papers.nips.cc/paper/5485-deep-convolutional-neural-network-for-image-deconvolution.pdf)|
2015|NIPS 2015|Shepard Convolutional Neural Networks [[pdf]](https://papers.nips.cc/paper/5774-shepard-convolutional-neural-networks.pdf) [[code]](https://github.com/jimmy-ren/vcnn_double-bladed/tree/master/applications/Shepard_CNN)|
2016|CVPR 2016|**Context Encoders: Feature Learning by Inpainting** [[pdf]](https://arxiv.org/abs/1604.07379) [[code]](https://github.com/pathak22/context-encoder)|
2016|SIGGRAPH 2016|High-resolution multi-scale neural texture synthesis [[pdf]](https://wxs.ca/research/multiscale-neural-synthesis/Snelgrove-multiscale-texture-synthesis.pdf)|
2017|CVPR 2017|Semantic image inpainting with deep generative models [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf) [[code]](https://github.com/moodoki/semantic_image_inpainting)|
2017|CVPR 2017|High-resolution image inpainting using multi-scale neural patch synthesis [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_High-Resolution_Image_Inpainting_CVPR_2017_paper.pdf) [[code]](https://github.com/leehomyc/Faster-High-Res-Neural-Inpainting)|
2017|CVPR 2017|Generative Face Completion [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Generative_Face_Completion_CVPR_2017_paper.pdf) [[code]](https://github.com/Yijunmaverick/GenerativeFaceCompletion)|
2017|SIGGRAPH 2017|**Globally and Locally Consistent Image Completion** [[pdf]](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/data/completion_sig2017.pdf) [[code]](https://github.com/satoshiiizuka/siggraph2017_inpainting)|
2018|CVPR 2018|**Generative Image Inpainting with Contextual Attention** [[pdf]](https://arxiv.org/abs/1801.07892) [[code]](https://github.com/JiahuiYu/generative_inpainting)|
2018|CVPR 2018|Natural and Effective Obfuscation by Head Inpainting [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Natural_and_Effective_CVPR_2018_paper.pdf)|
2018|CVPR 2018|Eye In-Painting With Exemplar Generative Adversarial Networks [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dolhansky_Eye_In-Painting_With_CVPR_2018_paper.pdf) [[code]](https://github.com/bdol/exemplar_gans)|
2018|CVPR 2018|UV-GAN: Adversarial Facial UV Map Completion for Pose-invariant Face Recognition [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_UV-GAN_Adversarial_Facial_CVPR_2018_paper.pdf)|
2018|CVPR 2018|Disentangling Structure and Aesthetics for Style-aware Image Completion [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gilbert_Disentangling_Structure_and_CVPR_2018_paper.pdf)|
2018|ECCV 2018|**Image Inpainting for Irregular Holes Using Partial Convolutions** [[pdf]](https://arxiv.org/pdf/1804.07723.pdf) [[code]](https://github.com/NVIDIA/partialconv)|
2018|ECCV 2018| Contextual-based Image Inpainting: Infer, Match, and Translate [[pdf]](https://arxiv.org/pdf/1711.08590.pdf)|
2018|ECCV 2018|Shift-Net: Image Inpainting via Deep Feature Rearrangement [[pdf]](https://arxiv.org/pdf/1801.09392v2.pdf) [[code]](https://github.com/Zhaoyi-Yan/Shift-Net)|
2018|NIPS 2018|Image Inpainting via Generative Multi-column Convolutional Neural Networks [[pdf]](https://arxiv.org/pdf/1810.08771.pdf) [[code]](https://github.com/shepnerd/inpainting_gmcnn)|
2018|TOG 2018|Faceshop: Deep sketch-based face image editing [[pdf]](https://arxiv.org/pdf/1804.08972.pdf)|
2018|ACM MM 2018|Structural inpainting [[pdf]](https://arxiv.org/pdf/1803.10348.pdf) |
2018|ACM MM 2018|Semantic Image Inpainting with Progressive Generative Networks [[pdf]](https://dl.acm.org/citation.cfm?id=3240625) [[code]](https://github.com/crashmoon/Progressive-Generative-Networks)|
2018|BMVC 2018|SPG-Net: Segmentation Prediction and Guidance Network for Image Inpainting [[pdf]](https://arxiv.org/pdf/1805.03356.pdf)|
2018|BMVC 2018|MC-GAN: Multi-conditional Generative Adversarial Network for Image Synthesi [[pdf]](https://arxiv.org/pdf/1805.01123.pdf) [[code]](https://github.com/HYOJINPARK/MC_GAN) |
2018|ACCV 2018|Face Completion iwht Semantic Knowledge and Collaborative Adversarial Learning [[pdf]](https://arxiv.org/pdf/1812.03252.pdf)|
2018|ICASSP 2018|Edge-Aware Context Encoder for Image Inpainting [[pdf]](http://mirlab.org/conference_papers/International_Conference/ICASSP%202018/pdfs/0003156.pdf)|
2018|ICPR 2018|Deep Structured Energy-Based Image Inpainting [[pdf]](https://arxiv.org/pdf/1801.07939.pdf) [[code]](https://github.com/cvlab-tohoku/DSEBImageInpainting)|
2018|AISTATS 2019|Probabilistic Semantic Inpainting with Pixel Constrained CNNs [[pdf]](https://arxiv.org/pdf/1810.03728.pdf)|
2019|ICRA 2019| Empty Cities: Image Inpainting for a Dynamic-Object-Invariant Space [[pdf]](https://arxiv.org/pdf/1809.10239.pdf)  | 
2019|AAAI 2019|Video Inpainting by Jointly Learning Temporal Structure and Spatial Details [[pdf]](https://arxiv.org/pdf/1806.08482.pdf)| Video
2019|CVPR 2019| Pluralistic Image Completion [[pdf]](https://arxiv.org/pdf/1903.04227.pdf) [[code]](https://github.com/lyndonzheng/Pluralistic) [[project]](http://www.chuanxiaz.com/publication/pluralistic/?tdsourcetag=s_pctim_aiomsg)|
2019|CVPR 2019| Deep Reinforcement Learning of Volume-guided Progressive View Inpainting for 3D Point Scene Completion from a Single Depth Image [[pdf]](https://arxiv.org/pdf/1903.04019.pdf)|
2019|CVPR 2019|Foreground-aware Image Inpainting [[pdf]](https://arxiv.org/pdf/1901.05945.pdf)  |
2019|CVPR 2019 |Privacy Protection in Street-View Panoramas using Depth and Multi-View Imagery [[pdf]](https://arxiv.org/pdf/1903.11532.pdf) |
2019|CVPR 2019|Learning Pyramid-Context Encoder Network for High-Quality Image Inpainting [[pdf]](https://arxiv.org/pdf/1904.07475.pdf) [[code]](https://github.com/researchmm/PEN-Net-for-Inpainting)
2019|CVPR 2019|Deep Flow-Guided Video Inpainting [[pdf]](https://arxiv.org/pdf/1905.02884.pdf) [[project]](https://nbei.github.io/video-inpainting.html)| Video
2019|CVPR 2019|Deep video inapinting [[pdf]](https://arxiv.org/pdf/1905.01639.pdf)|Video
2019|CVPR Workshop 2019 |VORNet: Spatio-temporally Consistent Video Inpainting for Object Removal [[pdf]](https://arxiv.org/pdf/1904.06726) | Video
2019|TNNLS 2019|PEPSI++: Fast and Lightweight Network for Image Inpainting [[pdf]](https://arxiv.org/pdf/1905.09010.pdf)| 
2019|ACM MM 2019| Progressive Image Inpainting with Full-Resolution Residual Network [[pdf]](https://arxiv.org/pdf/1907.10478.pdf)| 
2019|ICCV 2019|EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning [[pdf]](https://arxiv.org/pdf/1901.00212.pdf) [[code]](https://github.com/knazeri/edge-connect)|
2019|ICCV 2019|Coherent Semantic Attention for Image Inpainting [[pdf]](https://arxiv.org/pdf/1905.12384.pdf) [[code]](https://github.com/KumapowerLIU/CSA-inpainting)| 
2019|ICCV 2019|StructureFlow: Image Inpainting via Structure-aware Appearance Flow [[pdf]](https://arxiv.org/pdf/1908.03852.pdf) [[code]](https://github.com/RenYurui/StructureFlow) | 
2019|ICCV 2019| Image Inpainting with Learnable Bidirectional Attention Maps [[pdf]](https://arxiv.org/pdf/1909.00968.pdf) [[code]](https://github.com/Vious/LBAM_inpainting) |
2018|arXiv:1801.07632|High Resolution Face Completion with Multiple Controllable Attributes via Fully End-to-End Progressive Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1801.07632.pdf)|
2018|arXiv:1803.07422|Patch-Based Image Inpainting with Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1803.07422.pdf) [[code]](https://github.com/nashory/pggan-pytorch)|
2018|arXiv:1806.03589|Free-Form Image Inpainting with Gated Convolution [[pdf]](https://arxiv.org/pdf/1806.03589.pdf) [[project]](http://jiahuiyu.com/deepfill2/)|Gated Conv
2018|arXiv:1808.04432|X-GANs: Image Reconstruction Made Easy for Extreme Cases [[pdf]](https://arxiv.org/pdf/1808.04432.pdf)|
2018|arXiv:1811.03721|Learning Energy Based Inpainting for Optical Flow [[pdf]](https://arxiv.org/pdf/1811.03721.pdf)|
2018|arXiv:1811.07104|On Hallucinating Context and Background Pixels from a Face Mask using Multi-scale GANs [[pdf]](https://arxiv.org/pdf/1811.07104.pdf)|
2018|arXiv:1811.09012|Multi-View Inpainting for RGB-D Sequence [[pdf]](https://arxiv.org/pdf/1811.09012.pdf)|
2018|arXiv:1812.01458|Deep Inception Generative network for Cognitive Image Inpainting [[pdf]](https://arxiv.org/pdf/1812.01458.pdf)|
2019|arXiv:1901.03396|Detecting Overfitting of Deep Generative Networks via Latent Recovery [[pdf]](https://arxiv.org/pdf/1901.03396.pdf)|
2019|arXiv:1902.01096|Compatible and Diverse Fashion Image Inpainting [[pdf]](https://arxiv.org/pdf/1902.01096.pdf)  |
2019|arXiv:1902.06838|SC-FEGAN: Face Editing Generative Adversarial Network with User's Sketch and Color [[pdf]](https://arxiv.org/pdf/1902.06838.pdf) [[code]](https://github.com/JoYoungjoo/SC-FEGAN)  |
2019|arXiv:1902.09225|Harmonizing Maximum Likelihood with GANs for Multimodal Conditional Generation [[pdf]](https://arxiv.org/pdf/1902.09225.pdf)|   
2019|arXiv:1903.00450|Multi-Object Representation Learning with Iterative Variational Inference [[pdf]](https://arxiv.org/pdf/1903.00450.pdf) |    
2019|arXiv:1903.04842 |Unsupervised motion saliency map estimation based on optical flow inpainting [[pdf]](https://arxiv.org/pdf/1903.04842.pdf) |
2019|arXiv:1903.10885 |Learning Quadrangulated Patches For 3D Shape Processing [[pdf]](https://arxiv.org/pdf/1903.10885.pdf) |
2019|arXiv:1904.10247 |Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN [[pdf]](https://arxiv.org/pdf/1904.10247.pdf) |Video
2019|arXiv:1904.10795 |Graph-based Inpainting for 3D Dynamic Point Clouds [[pdf]](https://arxiv.org/pdf/1904.10795.pdf)| Point Cloud 
2019|arXiv:1905.02882 |Frame-Recurrent Video Inpainting by Robust Optical Flow Inference [[pdf]](https://arxiv.org/pdf/1905.02882.pdf)| Video
2019|arXiv:1905.02949 |Deep Blind Video Decaptioning by Temporal Aggregation and Recurrence [[pdf]](https://arxiv.org/pdf/1905.02949.pdf)| Video
2019|arXiv:1905.13066 |Align-and-Attend Network for Globally and Locally Coherent Video Inpainting [[pdf]](https://arxiv.org/pdf/1905.13066.pdf)| Video
2019|arXiv:1906.00884 |Fashion Editing with Multi-scale Attention Normalization [[pdf]](https://arxiv.org/pdf/1906.00884.pdf)| 



## Related

- [Globally and locally consistent image completion - Iizuka - SIGGRAPH 2017](https://arxivnote.ddlee.cn/Globally-and-locally-consistent-image-completion-SIGGRAPH.html)
- [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning - Nazeri - 2019 - PyTorch](https://arxivnote.ddlee.cn/EdgeConnect-Generative-Image-Inpainting-with-Adversarial-Edge-Learning-Nazeri.html)
- [Generative Image Inpainting with Contextual Attention - Yu - CVPR 2018 - TensorFlow](https://arxivnote.ddlee.cn/Generative-Image-Inpainting-with-Contextual-Attention-Yu-CVPR-TensorFlow.html)

