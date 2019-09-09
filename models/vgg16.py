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
#import torch.legacy.nn as nn_legacy
from torch.autograd import Variable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vgg_pytorch import vgg16 as model_vgg16

class Classifier(nn.Module):

  def __init__(self, pretrained='/npy/weights', OF_option='None', model_save_path='', test_model=''):

    super(Classifier, self).__init__()
    self.finetuning = pretrained
    self.OF_option = OF_option
    self.model_save_path = model_save_path

    if test_model=='':
      self._initialize_weights()
    else:
      self.model = model_vgg16(OF_option=self.OF_option, model_save_path=self.model_save_path, num_classes=22)

  def _initialize_weights(self):

    if 'emotionnet' in self.finetuning:
      mode='emotionnet'
      self.model = model_vgg16(pretrained=mode, OF_option=self.OF_option, model_save_path=self.model_save_path, num_classes=22)
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

    elif self.finetuning=='random':
      mode='RANDOM'
      self.model = model_vgg16(pretrained='', OF_option=self.OF_option, num_classes=1)      

    print("[OK] Weights initialized from %s"%(mode))

  def forward(self, image, OF=None):
    if OF is not None:
      x = self.model(image, OF=OF)
    else:
      x = self.model(image)
    return x      
