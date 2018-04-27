import sys
import pickle
import os
import cv2 as cv
import numpy as np
import random
import ipdb
import matlab.engine
import matlab
import time
import datetime
import tqdm
import glob
from shutil import copyfile

# from joblib import Parallel, delayed
# import multiprocessing
# num_cores = multiprocessing.cpu_count()

class Get_Faces():
  def __init__(self):
    pwd = os.getcwd()
    os.chdir('tools')
    from get_faces import __init__, face_from_file, imshow
    net_face = __init__()
    os.chdir(pwd)
    self.net = net_face
    self.face_from_file = face_from_file
    self.imshow = imshow        
    # import face_recognition
    # self.load_image_file = face_recognition.load_image_file
    # self.face_locations = face_recognition.face_locations #top, right, bottom, left
    # self.jit = 5
 
  # def imshow(self,image, name):
  #     import cv2
  #     cv2.startWindowThread()
  #     cv2.namedWindow('Image_'+str(name), cv2.WINDOW_NORMAL)
  #     cv2.imshow('Image_'+str(name),image)  


  def from_file(self, img_file, BBOX=False):
    # ipdb.set_trace()
    # img = self.load_image_file(img_file)
    # img_shape = img.shape[:-1]
    # img_jit = (np.array(img_shape)*self.jit/100).astype(np.uint8)
    # bbox = np.array(self.face_locations(img)[0])[[0,2,3,1]]
    # bbox_jit = [max(0,bbox[0]-img_jit[0]), min(img_shape[0], bbox[1]+img_jit[0]), \
    #             bbox[2], bbox[3]]
    # face = img[bbox_jit[0]:bbox_jit[1], bbox_jit[2]:bbox_jit[3]]
    # if BBOX:
    #     return face, bbox_jit
    # else:
    #     return face
    return self.face_from_file(self.net, img_file, BBOX=BBOX)


def display_time(start, end):
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  string_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
  print("Time elapsed: "+string_time)   
  print(datetime.datetime.now())  
  print(" ")

def get_resize(org_file, resized_file, img_size, OF=False):
  import skimage.transform
  import imageio
  if type(img_size)==int: img_size = [img_size, img_size]
  elif type(img_size)==list and len(img_size)==1: img_size = [img_file[0], img_size[0]]
  folder = os.path.dirname(resized_file)
  if not os.path.isdir(folder): os.makedirs(folder) 
  # ipdb.set_trace()
  ORDER = 1 if not OF else 0
  #1 bilinear - 0 nearest
  # if OF: ipdb.set_trace()
  imageio.imwrite(resized_file, \
    (skimage.transform.resize(imageio.imread(org_file), \
    (img_size[0], img_size[1]), order=ORDER)*255).astype(np.uint8))
  
if __name__ == '__main__':    
  import argparse
  import imageio

  parser = argparse.ArgumentParser(description='Txt file with path_to_image and 12 different AUs to LMDB')
  parser.add_argument('--img_size', type=int, default=256, help='size of the image to resize')
  parser.add_argument('--gpu', type=str, default='3', help='GPU device')
  parser.add_argument('--aligned', action='store_true', default=False)
  parser.add_argument('--OF', action='store_true', default=False)
  args = parser.parse_args()
  os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

  folder_root = '/home/afromero/datos/Databases/BP4D'
  txt_root = os.path.join(folder_root,'txt')
  org_root = os.path.join(folder_root,'Sequences')
  # face_root = '/home/afromero/Codes/ActionUnits/data/Faces/BP4D/Sequences'
  face_root = os.path.join(folder_root, 'Faces')
  bbox_root = os.path.join(folder_root, 'BBOX')
  algined_root = os.path.join(folder_root, 'Faces_aligned')
  algined_size_root = os.path.join(folder_root,'Faces_aligned_{}'.format(args.img_size))
  faces_size_root = os.path.join(folder_root,'Faces_{}'.format(args.img_size))

  if not os.path.isdir(txt_root): os.makedirs(txt_root)  
  if not os.path.isdir(face_root): os.makedirs(face_root)  
  if not os.path.isdir(bbox_root): os.makedirs(bbox_root)  
  if not os.path.isdir(algined_root): os.makedirs(algined_root)  
  if not os.path.isdir(algined_size_root): os.makedirs(algined_size_root)  

  pwd = os.getcwd()
  os.chdir('/home/afromero/Codes/Face_Alignment/MTCNN_face_detection_alignment/code/codes/MTCNNv2')
  future = matlab.engine.connect_matlab(async=True)
  eng = future.result()
  eng = matlab.engine.start_matlab()
  os.chdir(pwd)

  # ipdb.set_trace()
  img_files = sorted(glob.glob(org_root+'/*/*/*.jpg'))
  txt_file = os.path.join(txt_root, 'data.txt')
  f = open(txt_file, 'w')
  for im in img_files:
    f.writelines(os.path.abspath(im)+'\n')
  f.close()
  # ipdb.set_trace()

  org_files = [line.strip() for line in open(txt_file).readlines()]
  # ipdb.set_trace()
  Faces = Get_Faces()
  # face_files = []
  count = 0
  # ipdb.set_trace()        
  for org_file in tqdm.tqdm(org_files, total=len(org_files), desc='Extracting Faces'):
    file_name = '/'.join(org_file.split('/')[-3:])
    # folder_name = os.path.dirname(org_file)
    # face_name = file_name.split('.')[0]+'_Faces.'+file_name.split('.')[1]
    face_file = os.path.join(face_root, file_name)
    bbox_file = os.path.join(bbox_root, file_name).replace('jpg','txt')
    face_dir = os.path.dirname(face_file)
    bbox_dir = os.path.dirname(bbox_file)
    if not os.path.isdir(face_dir): os.makedirs(face_dir)
    if not os.path.isdir(bbox_dir): os.makedirs(bbox_dir)
    if not os.path.isdir(face_dir.replace(face_root, algined_root)): os.makedirs(face_dir.replace(face_root, algined_root))

    if os.path.isfile(face_file) and os.path.isfile(bbox_file):# and os.stat(bbox_file).st_size==0: 
      continue            
    img_face, bbox = Faces.from_file(org_file, BBOX=True)
      # ipdb.set_trace()
    # Faces.imshow(img_face)
    try: 
      if not os.path.isfile(face_file):# or True: 
        imageio.imwrite(face_file, img_face[:,:,::-1])
        # imageio.imwrite(face_file, img_face)
      f = open(bbox_file, 'w')
      for i in bbox: f.writelines(str(i)+'\n')
      f.close()
    except: 
      continue

    if args.OF:
      of_file = org_file.replace('Sequences', 'Sequences_Flow')
      of_face_file = face_file.replace('Faces', 'Faces_Flow')

      face_of_dir = os.path.dirname(of_face_file)
      if not os.path.isdir(face_of_dir): os.makedirs(face_of_dir)

      if not os.path.isfile(of_face_file):# or True:
        try: 
          of_face = imageio.imread(of_file)[bbox[0]:bbox[1],bbox[2]:bbox[3]]
          imageio.imwrite(of_face_file, of_face)
        except: 
          continue

      
  for org_file in tqdm.tqdm(org_files,  desc='Resizing Faces to %d'%(args.img_size)):
    rs_file = org_file.replace('Sequences', 'Faces')
    if not os.path.isfile(rs_file): continue
    file_name = '/'.join(rs_file.split('/')[-3:])
    resized_file = os.path.join(faces_size_root, file_name)
    if args.OF:
      resize_file_of = resized_file.replace('Faces', 'Faces_Flow')
      rs_file_of = rs_file.replace('Faces', 'Faces_Flow')
      if os.path.isfile(rs_file_of) and not os.path.isfile(resize_file_of):
        get_resize(rs_file_of, resize_file_of, args.img_size, OF=True)
    if not os.path.isfile(resized_file): get_resize(rs_file, resized_file, args.img_size)


