from __future__ import division
import ipdb
import inspect
import os
import time
import math
import glob
import numpy as np
from six.moves import xrange
import pickle
import sys
import config as cfg
import torch.nn as nn
import torch.legacy.nn as nn_legacy
from torch.autograd import Variable
import math
import torch
# torch.backends.cudnn.enabled=False

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from caffe_tensorflow import convert as caffe_tf
from vgg_pytorch import vgg16 as model_vgg16

class Classifier(nn.Module):

  def __init__(self, pretrained='/npy/weights', OF_option='None'):

    super(Classifier, self).__init__()
    self.finetuning = pretrained
    # self.std = np.array((0.229, 0.224, 0.225))

    self.OF_option = OF_option

    self._initialize_weights()

  def _initialize_weights(self):

    if 'emonet' in self.finetuning:
      self.model, filename = self.load_facexnet()
      mode=self.finetuning.upper()+' '+filename  

    elif 'emotionnet' in self.finetuning:
      mode='emotionnet'
      self.model = model_vgg16(pretrained=mode, OF_option=self.OF_option, num_classes=22)
      modules = self.model.modules()
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==22:
          w1 = m.weight.data[1:2].view(1,-1)
          b1 = torch.FloatTensor(np.array((m.bias.data[1])).reshape(-1))
      mod = list(self.model.classifier)
      mod.pop()
      if self.OF_option=='FC7': dim_fc7=4096*2
      else: dim_fc7 = 4096
      mod.append(torch.nn.Linear(dim_fc7,1))
      new_classifier = torch.nn.Sequential(*mod)
      self.model.classifier = new_classifier
      modules = self.model.modules()
      flag=False
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==1:
          m.weight.data = w1
          m.bias.data = b1   
          flag=True
      assert flag

    elif 'imagenet' in self.finetuning: 
      mode='ImageNet'
      self.model = model_vgg16(pretrained=mode, OF_option=self.OF_option, num_classes=1000)
      modules = self.model.modules()
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==1000:
          w1 = m.weight.data[1:2].view(1,-1)
          b1 = torch.FloatTensor(np.array((m.bias.data[1])).reshape(-1))
      mod = list(self.model.classifier)
      mod.pop()
      if self.OF_option=='FC7': dim_fc7=4096*2
      else: dim_fc7 = 4096
      mod.append(torch.nn.Linear(dim_fc7,1))
      new_classifier = torch.nn.Sequential(*mod)
      self.model.classifier = new_classifier
      flag=False
      modules = self.model.modules()
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==1:
          m.weight.data = w1
          m.bias.data = b1
          flag=True
      assert flag

    elif self.finetuning=='RANDOM':
      mode='RANDOM'
      self.model = model_vgg16(pretrained='', OF_option=self.OF_option, num_classes=1)      


    print("[OK] Weights initialized from %s"%(mode))
    
  def load_facexnet(self):
    npy_file = glob.glob(os.path.join('models/pretrained', self.finetuning, 'vgg16', '*.npy'))
    if len(npy_file)==0:
        caffemodel = glob.glob(os.path.join(self.finetuning, '*.caffemodel'))[0]
        npy_file = caffemodel.replace('.caffemodel', '.npy')
        self.caffemodel2npy(caffemodel, npy_file)    
    else: 
        npy_file = npy_file[0]
    print(" [*] Loading weights from: "+npy_file)
    params = np.load(npy_file, encoding='latin1').item()
    params={k.encode("utf-8"): v for k,v in params.iteritems()}
    #model.modules[31]=nn_legacy.View(-1,25088) #modules_caffe[32].weight.cpu().numpy().shape[1]
    #model.modules[-1].weight = model.modules[-1].weight[1].view(1,-1)
    #model.modules[-1].bias = torch.FloatTensor(np.array(model.modules[-1].bias[1]).reshape(-1))

    params_keys = sorted(params.keys())

    # ipdb.set_trace()
    conv_w = [params[m][0] for m in params_keys if 'conv' in m]
    conv_b = [params[m][1] for m in params_keys if 'conv' in m]
    fc_w = [params[m][0] for m in params_keys if 'fc' in m]
    fc_b = [params[m][1] for m in params_keys if 'fc' in m]

    if 'emonet' in self.finetuning or 'imagenet' in self.finetuning:
      # ipdb.set_trace()
      fc_w[-1] = fc_w[-1][:1].reshape(1,-1)
      fc_b[-1] = fc_b[-1][:1].reshape(-1)

    # ipdb.set_trace()

    model = model_vgg16(pretrained='', OF_option=self.OF_option, num_classes=1)
    for m in model.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data = torch.FloatTensor(conv_w.pop(0))
        m.bias.data = torch.FloatTensor(conv_b.pop(0))
      elif isinstance(m, nn.Linear):
        m.weight.data = torch.FloatTensor(fc_w.pop(0))
        m.bias.data = torch.FloatTensor(fc_b.pop(0))
# 
    # ipdb.set_trace()
        
    return model, npy_file

  def caffemodel2npy(self, caffemodel, npy_file):
    from caffe2npy import convert
    def_path = 'models/pretrained/aunet/vgg16/deploy_Test.pt'
    assert os.path.isfile(caffemodel), caffemodel+" must exist to finetune"
    # convert(def_path, caffemodel, npy_file) 
    # ipdb.set_trace()
    convert(def_path, caffemodel, npy_file, 'test')    
    # caffe_tf.convert(def_path, caffemodel, npy_file, npy_file.replace('npy','py'), 'test')

  def forward(self, image, OF=None):
    if OF is not None:
      x = self.model(image, OF=OF)
    else:
      x = self.model(image)
    return x      