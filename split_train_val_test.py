import os
import cv2 as cv
import sys
import glob
from os.path import join as opj
import ipdb
import shutil

root = 'data/MultiLabelAU/normal'
aligned = True

Train_files = sorted(glob.glob(opj(root, '*', 'Training.txt')))
Test_files = sorted(glob.glob(opj(root, '*', 'Test.txt')))


for file_txt in (Train_files):
  print(file_txt.split('/')[-2])
  lines = open(file_txt).readlines()
  train_subj = sorted(list(set([line.split('/')[-3] for line in lines])))
  val_subj = train_subj.pop(int(file_txt.split('/')[-2].split('_')[-1])+2)
  print("Train subj: %s"%(str(train_subj)))
  print("Val subj: %s"%(str(val_subj)))

  train_txt = file_txt.replace('Training', 'train')
  val_txt = train_txt.replace('train','val')
  test_file = file_txt.replace('Training','Test')

  if aligned:
    train_txt = train_txt.replace('normal', 'aligned')
    val_txt = val_txt.replace('normal', 'aligned')

  f_val = open(val_txt, 'w')
  f_train = open(train_txt, 'w')

  for i in xrange(len(lines)):
    if 'flip' in lines[i]: continue
    file_ = lines[i].replace('Codes/ActionUnits/data/Faces', 'datos/Databases')
    if 'BP4D_256' in file_: file_ = file_.replace('_256','')

    file_ = file_.replace('Sequences', 'Faces')
    if aligned: file_ = file_.replace('Faces', 'Faces_aligned')

    if val_subj in lines[i]:
      f_val.writelines(file_)
    else:
      f_train.writelines(file_)

  f_train.close()
  f_val.close()

  lines = open(test_file).readlines()
  test_subj = sorted(list(set([line.split('/')[-3] for line in lines])))
  print("Test subj: %s"%(str(test_subj)))
  if aligned:
    test_file = test_file.replace('normal', 'aligned')

  test_file = test_file.replace('Test','test')
  f_test = open(test_file, 'w')

  for i in xrange(len(lines)):
    if 'flip' in lines[i]: continue
    file_ = lines[i].replace('Codes/ActionUnits/data/Faces', 'datos/Databases')
    if 'BP4D_256' in file_: file_ = file_.replace('_256','')

    file_ = file_.replace('Sequences', 'Faces')
    if aligned: file_ = file_.replace('Faces', 'Faces_aligned')

    f_test.writelines(file_)
  f_test.close()
  print("Len Train: "+str(len(open(file_txt).readlines())))
  print("Len Val: "+str(len(open(val_txt).readlines())))
  print("Len Test: "+str(len(open(test_file).readlines())))
  print("")
