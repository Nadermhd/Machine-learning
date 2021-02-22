#!/usr/bin/env python2
# -*- coding: utf8 -*-

import os as os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import SimpleITK as sitk
import Models_3_out as M
import keras
import random
from keras.optimizers import Adam, SGD 

data_dir_training='/media/nader/WinD/Work/data/newsets/Brain'
data_dir_test='/media/nader/WinD/Work/data/newsets/Brain'

#####################################
#      Training and test Set
#####################################
class dataset_(object):
    
    def __init__(self, num_patient_tr, data_dir_training, num_patient_tst, data_dir_test, seed=123, size=(240, 240, 155)):
        self.data_dir_training = data_dir_training
        self.data_dir_test = data_dir_test
        self.num_patient_tr = num_patient_tr
        self.num_patient_tst = num_patient_tst
        self.size = size
        self.seed = seed

    def normalize(self, path_or, name_seg_o1):
        cMap1=sitk.ReadImage(os.path.join(path_or, name_seg_o1)) 
        cMap1=sitk.GetArrayFromImage(cMap1)
        
        me = np.mean(cMap1)
        std = np.std(cMap1)
        cMap1 = (cMap1 - me) / std
        
        min_v = np.min(cMap1)
        max_v = np.max(cMap1)
        cMap1 = (cMap1 - min_v) / (max_v - min_v)
        
        return cMap1
    
    def resize(self, arr, size_):
        from scipy.ndimage import zoom
        s, w, h = arr.shape[0], arr.shape[1], arr.shape[2] 
        return zoom(arr, (self.size[2]/s, self.size[0]/w, self.size[1]/h))

    def create_training_validating_data_abtImage(self, u): # abt channels
        
        #print('Loading training set....')
        training_mr=[]
        training_lb=[]
        patients_training=os.listdir(self.data_dir_training)
        patients_training.sort()
        
        path_or = os.path.join(self.data_dir_training)
        path_mr = os.path.join(path_or, 'imagesTr')
        path_seg = os.path.join(self.data_dir_training)
        path_lb = os.path.join(path_seg, 'labelsTr')
        
        files_mr = os.listdir(path_mr)
        files_lb = os.listdir(path_lb)
        # print(path_or, path_lb, files_mr)       

        name_org_mr = [fn for fn in files_mr if 'BRATS_'  in fn]
        name_org_mr = [fn for fn in name_org_mr if not '_BRATS_'  in fn]
        name_lbl_mr = [fn for fn in files_mr if 'BRATS_'  in fn]
        name_lbl_mr = [fn for fn in name_lbl_mr if not '_BRATS_'  in fn]
        
        name_org_mr.sort()
        name_lbl_mr.sort()
        
        c = list(zip(name_org_mr, name_lbl_mr))
        random.Random(self.seed).shuffle(c)
        name_org_mr, name_lbl_mr = zip(*c)
        
        print(u, name_org_mr[u])
        
        # mr
        normalized_org_mr = self.normalize(path_mr, name_org_mr[u])
        normalized_lbl_mr = self.normalize(path_lb, name_lbl_mr[u])
        
        #print('slice', np.shape(normalized_org_mr))
        
        normalized_org_mr = np.swapaxes(normalized_org_mr, 1, 2)
        normalized_org_mr = np.swapaxes(normalized_org_mr, 2, 3)
        normalized_org_mr = np.swapaxes(normalized_org_mr, 0, 3)
        
        normalized_lbl_mr = np.swapaxes(normalized_lbl_mr, 0, 1)
        normalized_lbl_mr = np.swapaxes(normalized_lbl_mr, 1, 2)
        normalized_lbl_mr = np.swapaxes(normalized_lbl_mr, 0, 2)
        
        # plt.subplot(2,1,1)
        # plt.imshow(normalized_org_mr[1, 100, ...])
        # plt.subplot(2,1,2)
        # plt.imshow(normalized_lbl_mr[100, ...])
        #print('slice', np.shape(normalized_org_mr))

        for ii in range(np.shape(normalized_org_mr)[3]):
            training_mr.append(normalized_org_mr[..., ii]) # (4, 240, 240, 155)
            training_lb.append(normalized_lbl_mr[..., ii]) # (240, 240, 155)
            
        return normalized_org_mr, normalized_lbl_mr, np.shape(normalized_org_mr)[3]
    
    #training_data = create_training_validating_data_abtImage(3, self.data_dir_training) 
  
    def create_testing_abtImage(self): # abt channels
        
        pass
    
    
    ##############################################################################
    #------------------------------- BATCH FUNCTIONS
    ##############################################################################
    def mini_batches(n,X,Y):
    
        Nsamples=np.shape(X)[0]
        I = np.arange(0,Nsamples)
        n_per_mini_batch = Nsamples//n   
        np.random.shuffle(I)
        return [[X[I[k*n_per_mini_batch:(k+1)*n_per_mini_batch]],Y[I[k*n_per_mini_batch:(k+1)*n_per_mini_batch]]] for k in range(0,n)]
    
    def DataSet_reshape(N, train_list, IMG_SIZE, SLICE_COUNT):
        XX=np.zeros((len(train_list), IMG_SIZE*IMG_SIZE*SLICE_COUNT*3))
        YY=np.zeros((len(train_list), 2))
        for i in range(len(train_list)):
            XX[i,:]=np.reshape(train_list[i][0], [IMG_SIZE*IMG_SIZE*SLICE_COUNT*3])
            YY[i,:]=np.reshape(train_list[i][1], [2])
        return XX,YY

d = dataset_(1, data_dir_training, 4, data_dir_test, seed=123, size=(240, 240, 155)) 
# tr, m = d.create_testing_abtImage()
im , lb, slices = d.create_training_validating_data_abtImage(0)
