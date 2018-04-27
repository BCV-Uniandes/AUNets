#!/usr/bin/env python

import os
import sys
sys.path.append('/home/afromero/caffe-master/python')
import numpy as np
import argparse
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

def fatal_error(msg):
  sys.stderr.write('%s\n' % msg)
  exit(-1)


def validate_arguments(args):
  if (args.data_output_path is not None) and (args.caffemodel is None):
    fatal_error('No input data path provided.')
  if (args.caffemodel is not None) and (args.data_output_path is None):
    fatal_error('No output data path provided.')
  if (args.data_output_path is None):
    fatal_error('No output path specified.')


def convert(def_path, caffemodel, data_output_path, phase):
  net = caffe.Net(def_path, caffemodel, getattr(caffe, phase.upper()))
  weights = net.params
  layers = weights.keys()
  npy_data = {}
  wb = [0,1]
  for l in layers:
    npy_data[l] = []
    for i in wb:
      npy_data[l].append(weights[l][i].data)
      if i==0: print("Saving "+l+", Weigths: "+str(weights[l][i].data.shape))
      elif i==1: print("Saving "+l+", Biases: "+str(weights[l][i].data.shape))
  np.save(data_output_path, npy_data)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('def_path', help='Model definition (.prototxt) path')
  parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
  parser.add_argument('--data-output-path', help='Converted data output path')
  parser.add_argument('--code-output-path', help='Save generated source to this path')
  parser.add_argument('-p',
                      '--phase',
                      default='test',
                      help='The phase to convert: test (default) or train')
  args = parser.parse_args()
  validate_arguments(args)
  convert(args.def_path, args.caffemodel, args.data_output_path, args.phase)
  #globals().update(vars(args))



if __name__ == '__main__':
    main()


#TO LOAD
#data_dict = np.load(npy_path, encoding='latin1').item()