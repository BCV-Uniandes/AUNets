import numpy as np
import imageio
import ipdb
import glob
import matplotlib.pyplot as plt
import os
import tqdm
import pickle

mode_data = 'normal'
txt_files = glob.glob('data/MultiLabelAU/{}/*/test.txt'.format(mode_data))
lines = []
mean = False
std  = True
for txt in txt_files:
  # ipdb.set_trace()
  lines.extend([i.split(' ')[0] for i in open(txt).readlines() if os.path.isfile(i.split(' ')[0])])
lines = list(set(sorted(lines)))
print("Calculating histogram from {} images...".format(len(lines)))

name = 'Faces_aligned' if 'aligned' in txt_files[0] else 'Faces'

img=np.zeros((256,256,3)).astype(np.float64)
sum=0
sum2=0
for line in tqdm.tqdm(lines, total=len(lines), desc='Calculating MEAN/STD', ncols=100, leave=False):
  line = line.replace(name, name+'_'+'256')
  img = imageio.imread(line).astype(np.uint64)
  sum += ( img / float(len(lines)) )
  sum2 += ( ((img)**2) / float(len(lines)) )

std = (np.sqrt( sum2 - (sum**2) )).astype(np.float64)
# ipdb.set_trace()

##MEAN
mean_img = 'data/face_{}_mean.jpg'.format(mode_data)
np.save(mean_img.replace('jpg','npy'), sum)
img = sum.astype(np.uint8)
imageio.imwrite(mean_img, img)

##STD
std_img = 'data/face_{}_std.jpg'.format(mode_data)
np.save(std_img.replace('jpg','npy'), std)
img = std.astype(np.uint8)
imageio.imwrite(std_img, img)
# ipdb.set_trace()