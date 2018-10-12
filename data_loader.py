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
import tqdm
import glob

class BP4D(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling = False, OF = False, verbose=False):
    # ipdb.set_trace()
    self.transform = transform
    self.mode = mode
    self.shuffling = shuffling
    self.image_size = image_size
    self.OF = OF
    self.verbose = verbose
    self.meta = '/home/afromero/datos/Databases/BP4D/'
    self.metaSSD = '/home/afromero/ssd2'
    # self.metaSSD = '../BP4D'
    if mode=='sample': mode='train'
    file_txt = os.path.join(metadata_path, mode+'.txt')
    self.lines = open(file_txt, 'r').readlines()

    if verbose: print ('Start preprocessing dataset (OF: %s): %s | file: %s!'%(str(OF), mode, file_txt))
    random.seed(1234)
    self.preprocess()
    if verbose: print ('Finished preprocessing dataset (OF: %s): %s!'%(str(OF), mode))
    
    self.num_data = len(self.filenames)

  def preprocess(self):
    self.filenames = []
    self.labels = []
    lines = [i.replace(self.meta, '').strip() for i in self.lines]
    if self.shuffling: random.shuffle(lines)   # random shuffling
    if self.verbose:
      iter_ = tqdm.tqdm(enumerate(lines), ncols=10, total=len(lines), desc='Preprocessing...')
    else:
      iter_ = enumerate(lines)
    for i, line in iter_:
      splits = line.split()
      filename = splits[0]
      if self.OF: 
        filename = filename.replace('Faces', 'Faces_Flow')

      label = [int(splits[1])]

      if self.mode=='sample' and 'Jitter' in filename: continue
      self.filenames.append(filename)
      self.labels.append(label)
    # ipdb.set_trace()

  def __getitem__(self, index):
    img_file = os.path.join(self.metaSSD, self.filenames[index])
    if not os.path.isfile(img_file): 
      print('%s not found'%(img_file))
      ipdb.set_trace()
      imageio.imwrite(img_file, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))    
    image = Image.open(img_file)
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data

class DEMO(Dataset):
  def __init__(self, path, transform, OF=False):
    # ipdb.set_trace()
    self.transform = transform
    if OF and os.path.isfile(path): raise TypeError("Cannot perform DEMO with Optical Flow for one single image. Please supply a directory with more than one image")

    if OF: 
      of_path = path+'_OF'
       
    if os.path.isfile(path):
      self.filenames = [path]

    elif os.path.isdir(path):
      self.filenames = glob.glob(path+'/*')

      if OF and len(glob.glob(of_path+'/*'))<=1: raise TypeError("Cannot perform DEMO with Optical Flow for one single image. Please supply a directory with more than one image")
      elif OF and not os.path.isdir(of_path): 
        os.makedirs(of_path)
        os.system('./generate_data/OF/broxDir --source {} --ziel {}'.format(path, of_path))
      if OF: 
        of_filenames = glob.glob(of_path+'/*')
        assert len(self.filenames)==len(of_filenames), "RGB and OF images must be the same"
        self.filenames = of_filenames
    self.filenames = sorted(self.filenames)
    self.num_data = len(self.filenames)

  def __getitem__(self, index):
    image = Image.open(self.filenames[index])
    return self.transform(image), torch.FloatTensor([0]), self.filenames[index]

  def __len__(self):
    return self.num_data

def get_loader(metadata_path, crop_size, image_size, batch_size, \
        mode='train', imagenet=False, OF=False, num_workers=0, verbose=False, demo=''):
  """Build and return data loader."""
  
  #ImageNet normalization
  if imagenet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0])

  transform = transforms.Compose([
      transforms.Resize((image_size, image_size), interpolation=Image.ANTIALIAS),
      transforms.ToTensor(),
      normalize,
      ])

  if demo:
    batch_size = 1
    dataset = DEMO(demo, transform, OF=OF)
  else:
    dataset = BP4D(image_size, metadata_path, transform, mode, \
              shuffling=mode=='train' or mode=='sample', OF=OF, verbose=verbose)

  data_loader = DataLoader(dataset=dataset,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers)
  return data_loader
