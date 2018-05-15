#!/usr/local/bin/ipython
import os
import argparse
from data_loader import get_loader
from torch.backends import cudnn
import torch
import glob
import math
import ipdb
import imageio
import numpy as np
import config as cfg
#CUDA_VISIBLE_DEVICES=0 ipython main.py -- --num_epochs 15 --batch_size 8 --image_size 256 --fold 0 --use_tensorboard --DYNAMIC_COLOR --CelebA_GAN


def main(config):
  # For fast training
  cudnn.benchmark = True

  # Create directories if not exist
  if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)
  if not os.path.exists(config.model_save_path):
    os.makedirs(config.model_save_path)

  # Data loader
  of_loader = None

  img_size = config.image_size

  rgb_loader = get_loader(config.metadata_path, img_size, img_size, config.batch_size, 
                    config.mode, num_workers=config.num_workers, OF=False, \
                   verbose=True, imagenet=config.finetuning=='imagenet', _255=config._255)   

  if config.OF: of_loader = get_loader(config.metadata_path, img_size,
                   img_size, config.batch_size, config.mode, num_workers=config.num_workers, OF=True, \
                   verbose=True, imagenet=config.finetuning=='imagenet', _255=config._255)

  # Solver
  from solver import Solver   
  solver = Solver(rgb_loader, config, of_loader=of_loader)

  if config.SHOW_MODEL: 
    solver.display_net()
    return

  if config.mode == 'train':
    solver.train()
    solver.test()
  elif config.mode == 'val':
    solver.val(load=True, init=True)
  elif config.mode == 'test':
    solver.test()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--image_size', type=int, default=224)
  parser.add_argument('--lr', type=float, default=0.0001)

  # Training settings
  parser.add_argument('--batch_size', type=int, default=118)  
  parser.add_argument('--dataset', type=str, default='BP4D', choices=['BP4D'])
  parser.add_argument('--num_epochs', type=int, default=12)
  parser.add_argument('--num_epochs_decay', type=int, default=13)
  parser.add_argument('--stop_training', type=int, default=2) #How many epochs after loss_val is not decreasing 
  parser.add_argument('--beta1', type=float, default=0.5)
  parser.add_argument('--beta2', type=float, default=0.999)
  parser.add_argument('--num_workers', type=int, default=4) 
  parser.add_argument('--HYDRA', action='store_true', default=False)
  parser.add_argument('--DELETE', action='store_true', default=False)
  parser.add_argument('--TEST_TXT', action='store_true', default=False)
  parser.add_argument('--TEST_PTH', action='store_true', default=False)

  # Optical Flow
  parser.add_argument('--OF', type=str, default='None', \
    choices=['None', 'Alone', 'Horizontal', 'Vertical', 'Channels', 'Conv', 'FC6', 'FC7'])

  # Test settings
  parser.add_argument('--test_model', type=str, default='')

  # Misc
  parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
  parser.add_argument('--use_tensorboard', action='store_true', default=False)
  parser.add_argument('--SHOW_MODEL', action='store_true', default=False)
  parser.add_argument('--GPU', type=str, default='3')

  # Path
  parser.add_argument('--metadata_path', type=str, default='./data')
  parser.add_argument('--log_path', type=str, default='./snapshot/logs')
  parser.add_argument('--model_save_path', type=str, default='./snapshot/models') 
  parser.add_argument('--results_path', type=str, default='./snapshot/results') 
  parser.add_argument('--fold', type=str, default='0')
  parser.add_argument('--mode_data', type=str, default='normal', choices=['normal', 'aligned'])  

  parser.add_argument('--AU', type=str, default='')
  parser.add_argument('--finetuning', type=str, default='emotionnet', choices=['emotionnet', 'imagenet', 'random'])  
  parser.add_argument('--pretrained_model', type=str, default='')   

  # Step size
  parser.add_argument('--log_step', type=int, default=2000)# tensorboard update

  config = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

  config = cfg.update_config(config)


  print(config)
  main(config)
