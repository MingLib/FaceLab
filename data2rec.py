"""
make .idx and .rec datafile
"""
import os, sys
import time
import pickle

import cv2
import tqdm
import mxnet as mx
import numpy as np
from PIL import Image

from Data.Vgg_Face2 import VggFace2, WarpAffine
from Data.Vgg_Face2.dataloader import k2landmark
from Data.CelebA import CelebA_Align


def alignVggface2rec():
    """
    read VggFace images and convert to .idx and .rec datafile
    """
    # get images path
    data = VggFace2("train", img_aligen=WarpAffine.similarity, crop_size=(112, 112))
    
    # create datafile lib
    working_dir = "/home/xuxiaoming/codesource/arcface_torch/Data/Vgg_Face2/data"
    fname_idx = "train.idx"
    fname_rec = "train.rec"
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    
    # convert
    for i in tqdm.trange(0,len(data)):
        
        filename = data.samples[i]
        cid = os.path.dirname(filename).strip()
        gtlabel = data.classid_to_label[cid]    # ground-truth label
        landmark = data.samples_landmark[i]     # landmark: five key points on the face
        fullpath = data.root/filename
        
        # read and align
        im = cv2.imread(str(fullpath))          # read image with opencv
        affine_grid = WarpAffine.similarity(k2landmark(landmark), data.reference_pts)
        im = cv2.warpAffine(im, affine_grid, (112, 112))
        
        # save to datafile
        header = mx.recordio.IRHeader(0, float(gtlabel), i, 0)
        s = mx.recordio.pack_img(header, im, quality=100, img_fmt=".jpeg")
        record.write_idx(i, s)
        
    record.close()

def alignCelebA2rec():
    """
    read CelebA images and convert to .idx and .rec datafile
    """
    data = CelebA_Align("train")    # image paths
    
    # create datafile lib
    working_dir = os.path.abspath("Data/CelebA/Img")
    fname_idx = "train.idx"
    fname_rec = "train.rec"
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    
    # convert
    reference_pts = WarpAffine.get_reference_facial_points((112, 112), 0, (0, 0), True)     # dist_landmark of affine transforme
    for i in tqdm.trange(0,len(data)):
        
        sample_idx = data.samples[i]
        img_path = data.imgs_root/data.imgs_path[sample_idx]    # image path
        gtlabel = data.targets[sample_idx]                      # ground-truth label
        landmark = data.landmarks[sample_idx]                   # image landmark
        
        # read and align
        im = cv2.imread(str(img_path))
        # double trans (for stability)
        affine_grid = WarpAffine.similarity(k2landmark(landmark), reference_pts)
        _ = WarpAffine.similarity(k2landmark(landmark), reference_pts)
        im = cv2.warpAffine(im, affine_grid, (112, 112))
        
        # save to datafile
        header = mx.recordio.IRHeader(0, float(gtlabel), i, 0)
        s = mx.recordio.pack_img(header, im, quality=100, img_fmt=".jpeg")
        record.write_idx(i, s)
        
    record.close()

def alignCelebA2bin():
    """
    affine the images and save to .bin file
    """
    # read images to list
    issame_list = []
    file_list = []
    with open(os.path.abspath("Data/CelebA/Eval/pairs.txt"), 'r') as pairfile:
        for pairinfo in pairfile.readlines():
            
            pairinfo = pairinfo.strip().split(' ')
            file0 = pairinfo[0]
            file1 = pairinfo[1]
            issame = int(pairinfo[2])
            
            file_list.append(file0)
            file_list.append(file1)
            issame_list.append(issame)
    
    # get landmark
    landmarks = []
    with open("Data/CelebA/Anno/list_landmarks_align_celeba.txt", 'r') as landmark_file:
        landmark_file.readline()
        landmark_file.readline()    # drop top-2 lines
        for line in landmark_file.readlines():
            line = line.strip().split(' ')
            
            # construct landmark for a single image
            temp_landmark = []
            for posnum in line[1:]: # drop top-1 element
                if len(posnum) > 1:
                    temp_landmark.append(float(posnum))
                    
            landmarks.append([line[0], temp_landmark])
            
    # affine transform
    i = 0 # counter
    celeba_bins = []
    reference_pts = WarpAffine.get_reference_facial_points((112, 112), 0, (0, 0), True) # dist_landmark of affine transforme
    for filename in file_list:
        
        # get image path
        idx = int(filename[:-4])-1
        fcheck, landmark = landmarks[idx]
        if filename != fcheck:
            print(1)
        image_path = os.path.join("Data/CelebA/Img/img_align_celeba", filename)

        # read and affine 
        im = cv2.imread(str(image_path))
        affine_grid = WarpAffine.similarity(k2landmark(landmark), reference_pts)
        _ = WarpAffine.similarity(k2landmark(landmark), reference_pts) # double check
        im = cv2.warpAffine(im, affine_grid, (112, 112))

        # convert to byte flow
        _, imarray = cv2.imencode(".jpg", im)
        _bin = imarray.tobytes()
        celeba_bins.append(_bin)
            
        i+=1
        if i%1000==0:
            print('loading data', i)

    # save to .bin file
    with open("celeba_test.bin", 'wb') as f:
        pickle.dump((celeba_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
        
def affinecrop():
    """
    align image and svae to normal pic fromat
    """
    # read images to list
    issame_list = []
    file_list = []
    with open(os.path.abspath("Data/CelebA/Eval/pairs.txt"), 'r') as pairfile:
        for pairinfo in pairfile.readlines():
            
            pairinfo = pairinfo.strip().split(' ')
            file0 = pairinfo[0]
            file1 = pairinfo[1]
            issame = int(pairinfo[2])

            file_list.append(file0)
            file_list.append(file1)
            issame_list.append(issame)
            
    # get landmark
    landmarks = []
    with open("Data/CelebA/Anno/list_landmarks_align_celeba.txt", 'r') as landmark_file:
        landmark_file.readline()
        landmark_file.readline()    # drop top-2 lines 
        for line in landmark_file.readlines():
            line = line.strip().split(' ')
            
            # construct landmark
            temp_landmark = []
            for posnum in line[1:]:
                if len(posnum) > 1:
                    temp_landmark.append(float(posnum))
                    
            landmarks.append([line[0], temp_landmark])
            
    # affine transform
    i = 0 # counter
    reference_pts = WarpAffine.get_reference_facial_points((112, 112), 0, (0, 0), True) # dist_landmark of affine transforme
    for filename in file_list:
        
        # get image path
        idx = int(filename[:-4])-1
        fcheck, landmark = landmarks[idx]
        if filename != fcheck:
            print(1)
        image_path = os.path.join("Data/CelebA/Img/img_align_celeba", filename)

        # read and affine
        im = cv2.imread(str(image_path))
        affine_grid = WarpAffine.similarity(k2landmark(landmark), reference_pts)
        _ = WarpAffine.similarity(k2landmark(landmark), reference_pts) # double check
        im = cv2.warpAffine(im, affine_grid, (112, 112))

        # save to pic format
        cv2.imwrite(os.path.join("Data/CelebA/Img/test", filename), im)
        
        i+=1
        if i%1000==0:
            print('loading data', i)

def makebinfile():
    """
    save images to .bin file
    """
    # read image paths and issame_list
    issame_list = []
    file_list = []
    with open(os.path.abspath("Data/CelebA/Eval/pairs.txt"), 'r') as pairfile:
        for pairinfo in pairfile.readlines():
            
            pairinfo = pairinfo.strip().split(' ')
            file0 = pairinfo[0]
            file1 = pairinfo[1]
            issame = int(pairinfo[2])

            file_list.append(file0)
            file_list.append(file1)
            issame_list.append(issame)
    
    # create bytes list
    celeba_bins = []
    for filename in file_list:
        image_path = os.path.join("Data/CelebA/Img/test", filename)
        
        # read and append byte flow of image to list
        with open(image_path, 'rb') as imfile:
            _bin = imfile.read()
            celeba_bins.append(_bin)
    
    # save to .bin file
    with open("celeba_test.bin", 'wb') as f:
        pickle.dump((celeba_bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
