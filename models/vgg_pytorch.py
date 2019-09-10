import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
import glob
"""
We provide pre-trained models, using the PyTorch :mod:`torch.utils.model_zoo`.
These can be constructed by passing ``pretrained=True``:
.. code:: python
  import torchvision.models as models
  vgg16 = models.vgg16(pretrained=True)
All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
where H and W are expected to be at least 224.
The images have to be loaded in to a range of [0, 1] and then normalized
using ``mean = [0.485, 0.456, 0.406]`` and ``std = [0.229, 0.224, 0.225]``.
You can use the following transform to normalize::

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])

An example of such normalization can be found in the imagenet example
`here <https://github.com/pytorch/examples/blob/\
    42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>`_
"""

__all__ = ['VGG', 'vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


# ========================================================================#
# ========================================================================#
# ========================================================================#
def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


# ========================================================================#
class VGG(nn.Module):
    def __init__(self, features, num_classes=2):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ========================================================================#
class VGG_ALONE(nn.Module):
    def __init__(self, features, num_classes=2):
        super(VGG_ALONE, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x, OF=None):
        OF = self.features(OF)
        OF = OF.view(OF.size(0), -1)
        OF = self.classifier(OF)
        return OF

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ========================================================================#
class VGG_IMAGE(nn.Module):
    def __init__(self, features, num_classes=2, OF_option='horizontal'):
        # img = torch.from_numpy(np.zeros((4,3,448,224), dtype=np.float32))
        super(VGG_IMAGE, self).__init__()
        self.OF_option = OF_option
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x, OF=None):
        if self.OF_option.lower() == 'horizontal':
            dim = 3
        else:
            dim = 2
        img_of = torch.cat([x, OF], dim=dim)
        x = self.features(img_of)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ========================================================================#
class VGG_CHANNELS(nn.Module):
    def __init__(self, features, num_classes=2):
        # img = torch.from_numpy(np.zeros((4,6,224,224), dtype=np.float32))
        super(VGG_CHANNELS, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x, OF=None):
        img_of = torch.cat([x, OF], dim=1)
        out = self.features(img_of)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ========================================================================#
class VGG_CONV(nn.Module):
    def __init__(self, features_rgb, features_of, num_classes=2):
        # img = torch.from_numpy(np.zeros((4,3,224,224), dtype=np.float32))
        super(VGG_CONV, self).__init__()
        self.features_rgb = features_rgb
        self.features_of = features_of
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x, OF=None):
        conv_rgb = self.features_rgb(x)
        conv_of = self.features_of(OF)

        conv_out = torch.cat([conv_rgb, conv_of], dim=1)
        conv_out = conv_out.view(conv_out.size(0), -1)

        out = self.classifier(conv_out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ========================================================================#
class VGG_FC6(nn.Module):
    def __init__(self, features_rgb, features_of, num_classes=2):
        super(VGG_FC6, self).__init__()
        self.features_rgb = features_rgb
        self.features_of = features_of
        self.classifier_rgb = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier_of = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x, OF=None):
        conv_rgb = self.features_rgb(x)
        conv_rgb = conv_rgb.view(conv_rgb.size(0), -1)
        fc6_rgb = self.classifier_rgb(conv_rgb)

        conv_of = self.features_of(OF)
        conv_of = conv_of.view(conv_of.size(0), -1)
        fc6_of = self.classifier_of(conv_of)

        fc_cat = torch.cat([fc6_rgb, fc6_of], dim=1)
        out = self.classifier(fc_cat)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ========================================================================#
class VGG_FC7(nn.Module):
    def __init__(self, features_rgb, features_of, num_classes=2):
        super(VGG_FC7, self).__init__()
        self.features_rgb = features_rgb
        self.features_of = features_of
        self.classifier_rgb = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier_of = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier = nn.Sequential(nn.Linear(8192, num_classes), )
        self._initialize_weights()

    def forward(self, x, OF=None):
        conv_rgb = self.features_rgb(x)
        conv_rgb = conv_rgb.view(conv_rgb.size(0), -1)
        fc7_rgb = self.classifier_rgb(conv_rgb)

        conv_of = self.features_of(OF)
        conv_of = conv_of.view(conv_of.size(0), -1)
        fc7_of = self.classifier_rgb(conv_of)

        fc_cat = torch.cat([fc7_rgb, fc7_of], dim=1)

        out = self.classifier(fc_cat)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ========================================================================#
# ========================================================================#
# ========================================================================#
# ========================================================================#


def vgg16(pretrained='', OF_option='None', model_save_path='', **kwargs):
    """VGG 16-layer model (configuration "D")
  Args:
    pretrained (str): If '', returns a model pre-trained on ImageNet
  """
    # ========================================================================#
    # ========================================================================#
    if pretrained == 'ImageNet':
        model_zoo_ = model_zoo.load_url(model_urls['vgg16'])
        model_zoo_ = {k.encode("utf-8"): v for k, v in model_zoo_.iteritems()}

    elif pretrained == 'emotionnet' and OF_option == 'None':
        emo_file = sorted(
            glob.glob('/home/afromero/datos2/EmoNet/snapshot/models/\
                    EmotionNet/normal/fold_all/Imagenet/*.pth'))[-1]
        model_zoo_ = torch.load(emo_file)
        # print("Finetuning from: "+emo_file)
        model_zoo_ = {
            k.replace('model.', ''): v
            for k, v in model_zoo_.iteritems()
        }

    elif pretrained == 'emotionnet' and OF_option != 'None':
        au_rgb_file = sorted(
            glob.glob(model_save_path.replace(OF_option, 'None') +
                      '/*.pth'))[-1]
        model_zoo_ = torch.load(au_rgb_file)
        # print("Finetuning from: "+os.path.abspath(au_rgb_file))
        model_zoo_ = {
            k.replace('model.', ''): v
            for k, v in model_zoo_.iteritems()
        }

    # ========================================================================#
    # ========================================================================#
    if OF_option == 'None':
        model = VGG(make_layers(cfg['D']), **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo_)

    # ========================================================================#
    elif OF_option == 'Alone':
        model = VGG_ALONE(make_layers(cfg['D']), **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo_)

    # ========================================================================#
    elif OF_option == 'Vertical' or OF_option == 'Horizontal':
        model = VGG_IMAGE(make_layers(cfg['D']), **kwargs)
        if pretrained:
            model_zoo_['classifier.0.weight'] = model_zoo_[
                'classifier.0.weight'].repeat(1, 2)
            model.load_state_dict(model_zoo_)

    # ========================================================================#
    elif OF_option == 'Channels':
        model = VGG_CHANNELS(make_layers(cfg['D'], in_channels=6), **kwargs)
        if pretrained:
            model_zoo_['features.0.weight'] = model_zoo_[
                'features.0.weight'].repeat(1, 2, 1, 1)
            model.load_state_dict(model_zoo_)

    # ========================================================================#
    elif OF_option == 'Conv':
        model = VGG_CONV(
            make_layers(cfg['D']), make_layers(cfg['D']), **kwargs)
        if pretrained:
            model_zoo_2 = {}
            model_zoo_2['classifier.0.weight'] = model_zoo_[
                'classifier.0.weight'].repeat(1, 2)
            conv_rgb_params = {
                k.replace('features', 'features_rgb'): v
                for k, v in model_zoo_.iteritems() if 'features' in k
            }
            model_zoo_2.update(conv_rgb_params)
            conv_of_params = {
                k.replace('features', 'features_of'): v
                for k, v in model_zoo_.iteritems() if 'features' in k
            }
            model_zoo_2.update(conv_of_params)
            fc_params = {
                k: v
                for k, v in model_zoo_.iteritems()
                if 'classifier' in k and 'classifier.0.weight' not in k
            }
            model_zoo_2.update(fc_params)
            model.load_state_dict(model_zoo_2)

    # ========================================================================#
    elif OF_option == 'FC6':
        model = VGG_FC6(make_layers(cfg['D']), make_layers(cfg['D']), **kwargs)
        if pretrained:
            model_zoo_2 = {}

            conv_rgb_params = {
                k.replace('features', 'features_rgb'): v
                for k, v in model_zoo_.iteritems() if 'features' in k
            }
            model_zoo_2.update(conv_rgb_params)
            fc_rgb_params = {
                k.replace('classifier', 'classifier_rgb'): v
                for k, v in model_zoo_.iteritems() if 'classifier.0' in k
            }
            model_zoo_2.update(fc_rgb_params)

            conv_of_params = {
                k.replace('features', 'features_of'): v
                for k, v in model_zoo_.iteritems() if 'features' in k
            }
            model_zoo_2.update(conv_of_params)
            fc_of_params = {
                k.replace('classifier', 'classifier_of'): v
                for k, v in model_zoo_.iteritems() if 'classifier.0' in k
            }
            model_zoo_2.update(fc_of_params)

            model_zoo_2['classifier.0.weight'] = model_zoo_[
                'classifier.3.weight'].repeat(1, 2)
            model_zoo_2['classifier.0.bias'] = model_zoo_['classifier.3.bias']
            model_zoo_2['classifier.3.weight'] = model_zoo_[
                'classifier.6.weight']
            model_zoo_2['classifier.3.bias'] = model_zoo_['classifier.6.bias']
            model.load_state_dict(model_zoo_2)

    # ========================================================================#
    elif OF_option == 'FC7':
        model = VGG_FC7(make_layers(cfg['D']), make_layers(cfg['D']), **kwargs)
        if pretrained:
            model_zoo_2 = {}
            conv_rgb_params = {
                k.replace('features', 'features_rgb'): v
                for k, v in model_zoo_.iteritems() if 'features' in k
            }
            model_zoo_2.update(conv_rgb_params)
            fc_rgb_params = {
                k.replace('classifier', 'classifier_rgb'): v
                for k, v in model_zoo_.iteritems()
                if 'classifier.0' in k or 'classifier.3' in k
            }
            model_zoo_2.update(fc_rgb_params)

            conv_of_params = {
                k.replace('features', 'features_of'): v
                for k, v in model_zoo_.iteritems() if 'features' in k
            }
            model_zoo_2.update(conv_of_params)
            fc_of_params = {
                k.replace('classifier', 'classifier_of'): v
                for k, v in model_zoo_.iteritems()
                if 'classifier.0' in k or 'classifier.3' in k
            }
            model_zoo_2.update(fc_of_params)

            model_zoo_2['classifier.0.weight'] = model_zoo_[
                'classifier.6.weight'].repeat(1, 2)
            model_zoo_2['classifier.0.bias'] = model_zoo_['classifier.6.bias']
            model.load_state_dict(model_zoo_2)

    return model
