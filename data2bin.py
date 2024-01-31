"""
make .bin datafile
"""
import os
import sys
import pickle
import argparse

import numpy as np
import mxnet as mx
from mxnet import ndarray as nd


def read_pairs(pairs_filename):
  """
  read face pairs info from .txt file
  
  .txt example
  - 1 path1 path2 0
  - 2 path1 path2 1
  """
  pairs = []
  with open(pairs_filename, 'r') as f:
    for line in f.readlines():
      pair = line.strip().split()
      pairs.append(pair)
  return np.array(pairs)

def get_paths(data_dir, pairs, file_ext):
  """
  get the path of each image in pairs file and label

  Returns:
      path_list: sequence the path of each pair of face iamges
      issame_list: squence the label of each pair
  """
  nrof_skipped_pairs = 0
  
  path_list = []
  issame_list = []
  for pair in pairs:
    
    # [path0, path1, label]
    if len(pair) == 3:
      path0 = os.path.join(data_dir, pair[0])
      path1 = os.path.join(data_dir, pair[1])
      if int(pair[2]) == 1:
        issame = True
      else:
        issame = False
    elif len(pair) == 4:
      path0 = os.path.join(data_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
      path1 = os.path.join(data_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
      issame = False
      
    # skip None image
    if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
      path_list += (path0, path1)
      issame_list.append(issame)
    else:
      print('not exists', path0, path1)
      nrof_skipped_pairs += 1
  
  if nrof_skipped_pairs > 0:
    print('Skipped %d image pairs' % nrof_skipped_pairs)
  return path_list, issame_list

def save2binfile(root_dir, dist_file):
  """
  save the squence of images and the issame_list to bin file
  """
  # get pairs and issame_list
  data_pairs = read_pairs(os.path.join(root_dir, 'pairs.txt'))
  data_paths, issame_list = get_paths(root_dir, data_pairs, 'jpg')
  print(len(data_paths))
  print(len(issame_list))
  
  # read images
  lfw_bins = []
  #lfw_data = nd.empty((len(lfw_paths), 3, image_size[0], image_size[1]))
  i = 0 # counter
  for path in data_paths:
    with open(path, 'rb') as fin:
      _bin = fin.read()
      lfw_bins.append(_bin)
      #img = mx.image.imdecode(_bin)
      #img = nd.transpose(img, axes=(2, 0, 1))
      #lfw_data[i][:] = img
      i+=1
      if i%1000==0:
        print('loading data', i)
        
  # save
  with open(dist_file, 'wb') as f:
    pickle.dump((lfw_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)





