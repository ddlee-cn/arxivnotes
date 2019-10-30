---
title: Specifying Object Attributes and Relations in Interactive Scene Generation
tag:
- Image Synthesis
- Interactive
---

## Info

- Title: Specifying Object Attributes and Relations in Interactive Scene Generation
- Task: Image Generation
- Author: Oron Ashual, Lior Wolf
- Date: Sep. 2019
- Arxiv: [1909.05379](https://arxiv.org/abs/1909.05379)
- Published: ICCV 2019(Best Paper Award Honorable Mentions)

## Abstract

We introduce a method for the generation of images from an input scene graph. The method separates between a layout embedding and an appearance embedding. The dual embedding leads to generated images that better match the scene graph, have higher visual quality, and support more complex scene graphs. In addition, the embedding scheme supports multiple and diverse output images per scene graph, which can be further controlled by the user. We demonstrate two modes of per-object control: (i) importing elements from other images, and (ii) navigation in the object space, by selecting an appearance archetype.




## Motivation & Design

![](https://github.com/ashual/scene_generation/blob/master/images/scene_generation.png)

(top row) the schematic illustration panel of the user interface, in which the user arranges the desired objects. (2nd row) the scene graph that is inferred automatically based on this layout. (3rd row) the layout that is created from the scene graph. (bottom row) the generated image. Legend for the GUI colors in the top row: purple – adding an object, green – resizing it, red – replacing its appearance. (a) A simple layout with a sky object, a tree and a grass object. All object appearances are initialized to a random archetype appearance. (b) A giraffe is added. (c) The giraffe is enlarged. (d) The appearance of the sky is changed to a different archetype. (e) A small sheep is added. (f) An airplane is added. (g) The tree is enlarged.


![](https://github.com/ashual/scene_generation/blob/master/images/arch.png)

The architecture of our composite network, including the subnetworks G, M, B, A, R, and the process of creating the layout tensor t. The scene graph is passed to the network G to create the layout embedding ui of each object. The bounding box bi is created from this embedding, using network B. A random vector zi is concatenated to ui , and the network M computes the mask mi . The appearance information, as encoded by the network A, is then added to create the tensor t with c + d5 channels, c being the number of classes. The autoencoder R generates the final image p from this tensor.


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


[PyTorch](https://github.com/ashual/scene_generation)


### Model

```python
class Model(nn.Module):
    def __init__(self, ...):
        super(Model, self).__init__()
        
        self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        box_net_dim = 4
        self.box_dim = box_dim
        box_net_layers = [self.box_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

        self.g_mask_dim = gconv_dim + mask_noise_dim
        self.mask_net = mask_net(self.g_mask_dim, mask_size)

        self.repr_input = self.g_mask_dim
        rep_size = rep_size
        rep_hidden_size = 64
        repr_layers = [self.repr_input, rep_hidden_size, rep_size]
        self.repr_net = build_mlp(repr_layers, batch_norm=mlp_normalization)

        self.image_encoder = AppearanceEncoder(**appearance_encoder_kwargs)

        self.layout_to_image = define_G(netG_input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm)

    def forward(self, gt_imgs, objs, triples, obj_to_img, boxes_gt=None, masks_gt=None, attributes=None,
                gt_train=False, test_mode=False, use_gt_box=False, features=None):
        O, T = objs.size(0), triples.size(0)
        obj_vecs, pred_vecs = self.scene_graph_to_vectors(objs, triples, attributes)

        box_vecs, mask_vecs, scene_layout_vecs, wrong_layout_vecs = \
            self.create_components_vecs(gt_imgs, boxes_gt, obj_to_img, objs, obj_vecs, gt_train, features)

        # Generate Boxes
        boxes_pred = self.box_net(box_vecs)

        # Generate Masks
        mask_scores = self.mask_net(mask_vecs.view(O, -1, 1, 1))
        masks_pred = mask_scores.squeeze(1).sigmoid()

        H, W = self.image_size

        return imgs_pred, boxes_pred, masks_pred, gt_layout, pred_layout, wrong_layout

    def scene_graph_to_vectors(self, objs, triples, attributes):
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)
        if self.use_attributes:
            obj_vecs = torch.cat([obj_vecs, attributes], dim=1)

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        return obj_vecs, pred_vecs

    def create_components_vecs(self, imgs, boxes, obj_to_img, objs, obj_vecs, gt_train, features):
        O = objs.size(0)
        box_vecs = obj_vecs
        mask_vecs = obj_vecs
        layout_noise = torch.randn((1, self.mask_noise_dim), dtype=mask_vecs.dtype, device=mask_vecs.device) \
            .repeat((O, 1)) \
            .view(O, self.mask_noise_dim)
        mask_vecs = torch.cat([mask_vecs, layout_noise], dim=1)

        # create encoding
        crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
        obj_repr = self.repr_net(self.image_encoder(crops))

        # Only in inference time
        if features is not None:
            for ind, feature in enumerate(features):
                if feature is not None:
                    obj_repr[ind, :] = feature
        # create one-hot vector for label map
        one_hot_size = (O, self.num_objs)
        one_hot_obj = torch.zeros(one_hot_size, dtype=obj_repr.dtype, device=obj_repr.device)
        one_hot_obj = one_hot_obj.scatter_(1, objs.view(-1, 1).long(), 1.0)
        layout_vecs = torch.cat([one_hot_obj, obj_repr], dim=1)

        wrong_objs_rep = self.fake_pool.query(objs, obj_repr)
        wrong_layout_vecs = torch.cat([one_hot_obj, wrong_objs_rep], dim=1)
        return box_vecs, mask_vecs, layout_vecs, wrong_layout_vecs

    def encode_scene_graphs(self, scene_graphs, rand=False):
        """
        Encode one or more scene graphs using this model's vocabulary. Inputs to
        this method are scene graphs represented as dictionaries like the following:
        {
          "objects": ["cat", "dog", "sky"],
          "relationships": [
            [0, "next to", 1],
            [0, "beneath", 2],
            [2, "above", 1],
          ]
        }
        This scene graph has three relationshps: cat next to dog, cat beneath sky,
        and sky above dog.
        Inputs:
        - scene_graphs: A dictionary giving a single scene graph, or a list of
          dictionaries giving a sequence of scene graphs.
        Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
        same semantics as self.forward. The returned LongTensors will be on the
        same device as the model parameters.
        """
        if isinstance(scene_graphs, dict):
            # We just got a single scene graph, so promote it to a list
            scene_graphs = [scene_graphs]
        device = next(self.parameters()).device
        objs, triples, obj_to_img = [], [], []
        all_attributes = []
        all_features = []
        obj_offset = 0
        for i, sg in enumerate(scene_graphs):
            attributes = torch.zeros([len(sg['objects']) + 1, 25 + 10], dtype=torch.float, device=device)
            # Insert dummy __image__ object and __in_image__ relationships
            sg['objects'].append('__image__')
            sg['features'].append(0)
            image_idx = len(sg['objects']) - 1
            for j in range(image_idx):
                sg['relationships'].append([j, '__in_image__', image_idx])

            for obj in sg['objects']:
                obj_idx = self.vocab['object_to_idx'][str(self.vocab['object_name_to_idx'][obj])]
                if obj_idx is None:
                    raise ValueError('Object "%s" not in vocab' % obj)
                objs.append(obj_idx)
                obj_to_img.append(i)
            if self.features is not None:
                for obj_name, feat_num in zip(objs, sg['features']):
                    feat = torch.from_numpy(self.features[obj_name][min(feat_num, 9), :]).type(torch.float32).to(
                        device) if feat_num != -1 else None
                    all_features.append(feat)
            for s, p, o in sg['relationships']:
                pred_idx = self.vocab['pred_name_to_idx'].get(p, None)
                if pred_idx is None:
                    raise ValueError('Relationship "%s" not in vocab' % p)
                triples.append([s + obj_offset, pred_idx, o + obj_offset])
            for i, size_attr in enumerate(sg['attributes']['size']):
                attributes[i, size_attr] = 1
            # in image size
            attributes[-1, 9] = 1
            for i, location_attr in enumerate(sg['attributes']['location']):
                attributes[i, location_attr + 10] = 1
            # in image location
            attributes[-1, 12 + 10] = 1
            obj_offset += len(sg['objects'])
            all_attributes.append(attributes)
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
        attributes = torch.cat(all_attributes)
        features = all_features
        return objs, triples, obj_to_img, attributes, features
```


### GraphTripleConvNet

```python
class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, input_dim, attributes_dim=0, output_dim=None, hidden_dim=512,
                 pooling='avg', mlp_normalization='none'):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [3 * input_dim + 2 * attributes_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
        self.net2.apply(_init_weights)

    def forward(self, obj_vecs, pred_vecs, edges):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
        Outputs:
        - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (T, Din)
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        # Get current vectors for triples; shape is (T, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
        # p vecs have shape (T, Dout)
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]
        new_o_vecs = new_t_vecs[:, (H + Dout):(2 * H + Dout)]

        # Allocate space for pooled object vectors of shape (O, H)
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (T, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (O, Dout)
        new_obj_vecs = self.net2(pooled_obj_vecs)

        return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
                 mlp_normalization='none'):
        super(GraphTripleConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'mlp_normalization': mlp_normalization,
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs
```

### Genrators

```python
def mask_net(dim, mask_size):
    output_dim = 1
    layers, cur_size = [], 1
    while cur_size < mask_size:
        layers.append(Interpolate(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(dim))
        layers.append(nn.ReLU())
        cur_size *= 2
    if cur_size != mask_size:
        raise ValueError('Mask size must be a power of 2')
    layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)


class AppearanceEncoder(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 padding='same', vecs_size=1024, pooling='avg'):
        super(AppearanceEncoder, self).__init__()
        self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, channels = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(channels, vecs_size))

    def forward(self, crops):
        return self.cnn(crops)


def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance'):
    norm_layer = get_norm_layer(norm_type=norm)
    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    assert (torch.cuda.is_available())
    netG.cuda()
    netG.apply(weights_init)
    return netG

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
```

### Discriminators

```python
class AcDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu', padding='same', pooling='avg'):
        super(AcDiscriminator, self).__init__()
        self.vocab = vocab

        cnn_kwargs = {
            'arch': arch,
            'normalization': normalization,
            'activation': activation,
            'pooling': pooling,
            'padding': padding,
        }
        cnn, D = build_cnn(**cnn_kwargs)
        self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
        num_objects = len(vocab['object_to_idx'])

        self.real_classifier = nn.Linear(1024, 1)
        self.obj_classifier = nn.Linear(1024, num_objects)

    def forward(self, x, y):
        if x.dim() == 3:
            x = x[:, None]
        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)
        obj_scores = self.obj_classifier(vecs)
        ac_loss = F.cross_entropy(obj_scores, y)
        return real_scores, ac_loss


class AcCropDiscriminator(nn.Module):
    def __init__(self, vocab, arch, normalization='none', activation='relu',
                 object_size=64, padding='same', pooling='avg'):
        super(AcCropDiscriminator, self).__init__()
        self.vocab = vocab
        self.discriminator = AcDiscriminator(vocab, arch, normalization,
                                             activation, padding, pooling)
        self.object_size = object_size

    def forward(self, imgs, objs, boxes, obj_to_img):
        crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
        real_scores, ac_loss = self.discriminator(crops, objs)
        return real_scores, ac_loss, crops


def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D)
    # print(netD)
    assert (torch.cuda.is_available())
    netD.cuda()
    netD.apply(weights_init)
    return netD


def define_mask_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1,
                  num_objects=None):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleMaskDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D,
                                       num_objects)
    assert (torch.cuda.is_available())
    netD.cuda()
    netD.apply(weights_init)
    return netD


class MultiscaleMaskDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, num_objects=None):
        super(MultiscaleMaskDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerMaskDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, num_objects)
            for j in range(n_layers + 2):
                setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input, cond):
        result = [input]
        for i in range(len(model) - 2):
            # print(result[-1].shape)
            result.append(model[i](result[-1]))

        a, b, c, d = result[-1].shape
        cond = cond.view(a, -1, 1, 1).expand(-1, -1, c, d)
        concat = torch.cat([result[-1], cond], dim=1)
        result.append(model[len(model) - 2](concat))
        result.append(model[len(model) - 1](result[-1]))
        return result[1:]

    def forward(self, input, cond):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                     range(self.n_layers + 2)]
            result.append(self.singleD_forward(model, input_downsampled, cond))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

```

### Training Process

```python
for batch in train_loader:
    t += 1
    imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img, attributes = batch

    use_gt = random.randint(0, 1) != 0
    if not use_gt:
        attributes = torch.zeros_like(attributes)
    model_out = trainer.model(imgs, objs, triples, obj_to_img,
                              boxes_gt=boxes, masks_gt=masks, attributes=attributes, gt_train=use_gt)
    imgs_pred, boxes_pred, masks_pred, layout, layout_pred, layout_wrong = model_out

    layout_one_hot = layout[:, :trainer.num_obj, :, :]
    layout_pred_one_hot = layout_pred[:, :trainer.num_obj, :, :]

    trainer.train_generator(imgs, imgs_pred, masks, masks_pred, layout,
                            objs, boxes, boxes_pred, obj_to_img, use_gt)
    trainer.train_mask_discriminator(masks, masks_pred_detach, objs)
    trainer.train_obj_discriminator(imgs, imgs_pred_detach, objs, boxes, boxes_pred_detach, obj_to_img)
    trainer.train_image_discriminator(imgs, imgs_pred_detach, layout_detach, layout_wrong_detach)



class Trainer:
    def __init__(self, ...):

    def train_generator(self, imgs, imgs_pred, masks, masks_pred, layout,
                        objs, boxes, boxes_pred, obj_to_img, use_gt):
        args = self.args
        self.generator_losses = LossManager()

        if use_gt:
            if args.l1_pixel_loss_weight > 0:
                l1_pixel_loss = F.l1_loss(imgs_pred, imgs)
                self.generator_losses.add_loss(l1_pixel_loss, 'L1_pixel_loss', args.l1_pixel_loss_weight)

            loss_bbox = F.mse_loss(boxes_pred, boxes)
            self.generator_losses.add_loss(loss_bbox, 'bbox_pred', args.bbox_pred_loss_weight)

        # VGG feature matching loss
        if self.criterionVGG is not None:
            loss_G_VGG = self.criterionVGG(imgs_pred, imgs)
            self.generator_losses.add_loss(loss_G_VGG, 'g_vgg', args.vgg_features_weight)

        scores_fake, ac_loss, g_fake_crops = self.obj_discriminator(imgs_pred, objs, boxes, obj_to_img)
        self.generator_losses.add_loss(ac_loss, 'ac_loss', args.ac_loss_weight)
        weight = args.d_obj_weight
        self.generator_losses.add_loss(self.gan_g_loss(scores_fake), 'g_gan_obj_loss', weight)

        if self.mask_discriminator is not None:
            O, _, mask_size = masks_pred.shape
            one_hot_size = (O, self.num_obj)
            one_hot_obj = torch.zeros(one_hot_size, dtype=masks_pred.dtype, device=masks_pred.device)
            one_hot_obj = one_hot_obj.scatter_(1, objs.view(-1, 1).long(), 1.0)

            scores_fake = self.mask_discriminator(masks_pred.unsqueeze(1), one_hot_obj)
            mask_loss = self.criterionGAN(scores_fake, True)
            self.generator_losses.add_loss(mask_loss, 'g_gan_mask_obj_loss', args.d_mask_weight)

            # GAN feature matching loss
            if args.d_mask_features_weight > 0:
                scores_real = self.mask_discriminator(masks.float().unsqueeze(1), one_hot_obj)
                loss_mask_feat = self.calculate_features_loss(scores_fake, scores_real)
                self.generator_losses.add_loss(loss_mask_feat, 'g_mask_features_loss', args.d_mask_features_weight)

        if self.netD is not None:
            # Train textures
            pred_real = self.netD.forward(torch.cat((layout, imgs), dim=1))

            # Train image generation
            match_layout = layout.detach()
            img_pred_fake = self.netD.forward(torch.cat((match_layout, imgs_pred), dim=1))
            g_gan_img_loss = self.criterionGAN(img_pred_fake, True)
            self.generator_losses.add_loss(g_gan_img_loss, 'g_gan_img_loss', args.d_img_weight)

            if args.d_img_features_weight > 0:
                loss_g_gan_feat_img = self.calculate_features_loss(img_pred_fake, pred_real)
                self.generator_losses.add_loss(loss_g_gan_feat_img,
                                               'g_gan_features_loss_img', args.d_img_features_weight)

        self.generator_losses.all_losses['total_loss'] = self.generator_losses.total_loss.item()

        self.optimizer.zero_grad()
        self.generator_losses.total_loss.backward()
        self.optimizer.step()

    def train_obj_discriminator(self, imgs, imgs_pred, objs, boxes, boxes_pred, obj_to_img):
        if self.obj_discriminator is not None:
            self.d_obj_losses = d_obj_losses = LossManager()
            scores_fake, ac_loss_fake, self.d_fake_crops = self.obj_discriminator(imgs_pred, objs, boxes_pred,
                                                                                  obj_to_img)
            scores_real, ac_loss_real, self.d_real_crops = self.obj_discriminator(imgs, objs, boxes, obj_to_img)

            d_obj_gan_loss = self.gan_d_loss(scores_real, scores_fake)
            d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss', 0.5)
            d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
            d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

            self.optimizer_d_obj.zero_grad()
            d_obj_losses.total_loss.backward()
            self.optimizer_d_obj.step()

    def train_mask_discriminator(self, masks, masks_pred, objs):
        if self.mask_discriminator is not None:
            self.d_mask_losses = d_mask_losses = LossManager()

            O, _, mask_size = masks_pred.shape
            one_hot_size = (O, self.num_obj)
            one_hot_obj = torch.zeros(one_hot_size, dtype=masks_pred.dtype, device=masks_pred.device)
            one_hot_obj = one_hot_obj.scatter_(1, objs.view(-1, 1).long(), 1.0)

            scores_fake = self.mask_discriminator(masks_pred.unsqueeze(1), one_hot_obj)
            scores_real = self.mask_discriminator(masks.float().unsqueeze(1), one_hot_obj)

            fake_loss = self.criterionGAN(scores_fake, False)
            real_loss = self.criterionGAN(scores_real, True)
            d_mask_losses.add_loss(fake_loss, 'fake_loss', 0.5)
            d_mask_losses.add_loss(real_loss, 'real_loss', 0.5)

            self.optimizer_d_mask.zero_grad()
            d_mask_losses.total_loss.backward()
            self.optimizer_d_mask.step()

    def train_image_discriminator(self, imgs, imgs_pred, layout, layout_wrong):
        if self.netD is not None:
            self.d_img_losses = d_img_losses = LossManager()
            # Fake Detection and Loss
            alpha = (1 / 2) * (.5)

            # Fake images, Real layout
            pred_fake_pool_img = self.discriminate(layout, imgs_pred)
            loss_d_fake_img = self.criterionGAN(pred_fake_pool_img, False)
            d_img_losses.add_loss(loss_d_fake_img, 'fake_image_loss', alpha)

            # Real images, Right layout Wrong textures
            pred_wrong_pool_img = self.discriminate(layout_wrong, imgs)
            loss_d_wrong_texture = self.criterionGAN(pred_wrong_pool_img, False)
            d_img_losses.add_loss(loss_d_wrong_texture, 'wrong_texture_loss', alpha)

            # Real Detection and Loss
            pred_real = self.discriminate(layout, imgs)
            loss_D_real = self.criterionGAN(pred_real, True)
            d_img_losses.add_loss(loss_D_real, 'd_img_gan_real_loss', 0.5)

            self.optimizer_d_img.zero_grad()
            d_img_losses.total_loss.backward()
            self.optimizer_d_img.step()
```

## Related

