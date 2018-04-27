import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import ipdb
import numpy as np
import imageio
import scipy.misc


class BP4D(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling = False, OF = False):
    # ipdb.set_trace()
    self.transform = transform
    self.mode = mode
    self.shuffling = shuffling
    self.image_size = image_size
    self.OF = OF
    self.lines = open(os.path.join(metadata_path, mode+'.txt'), 'r').readlines()

    print ('Start preprocessing dataset (OF: %s): %s!'%(str(OF), mode))
    random.seed(1234)
    self.preprocess()
    print ('Finished preprocessing dataset (OF: %s): %s!'%(str(OF), mode))
    
    self.num_data = len(self.filenames)

  def preprocess(self):
    self.filenames = []
    self.labels = []
    lines = [i.strip() for i in self.lines]
    if self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):
      splits = line.split()
      filename = splits[0]
      if self.OF: 
        filename = filename.replace('Faces', 'Faces_Flow')
        if not os.path.isfile(filename): 
          imageio.imwrite(filename, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))

      label = [int(splits[1])]

      self.filenames.append(filename)
      self.labels.append(label)
    # ipdb.set_trace()

  def __getitem__(self, index):
    image = Image.open(self.filenames[index])
    label = self.labels[index]
    # ipdb.set_trace()
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data

def get_loader(metadata_path, crop_size, image_size, batch_size, \
        mode='train', OF=False, num_workers=0):
  """Build and return data loader."""
  
  #ImageNet normalization
  # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0])

  if mode == 'train':
    transform = transforms.Compose([
      # transforms.CenterCrop(crop_size),
      transforms.Resize((image_size, image_size), interpolation=Image.ANTIALIAS),   
      transforms.ToTensor(),
      normalize,
      # transforms.Normalize(mean, std),
      ])  


  else:
    transform = transforms.Compose([
      # transforms.CenterCrop(crop_size),
      transforms.Resize((image_size, image_size), interpolation=Image.ANTIALIAS),
      # transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
      transforms.ToTensor(),
      # transforms.Normalize(mean, std),
      normalize,
      ])

  dataset = BP4D(image_size, metadata_path, transform, mode, \
              shuffling=mode=='train', OF=OF)

  data_loader = DataLoader(dataset=dataset,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers)
  return data_loader
