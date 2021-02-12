# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:19:00 2019

@author: aldojn
"""

import os as os
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import SimpleITK as sitk

data_dir_training='/training'
data_dir_test='test_new_2'

patients_training=os.listdir(data_dir_training)
patients_training.sort()


cost_plot=[]
loss_plot=[]
accu_plot=[]

IMG_SIZE = 64
SLICE_COUNT = 1


#####################################
#       Reshapping 3D-Vector
#####################################

def reshape_3DToVector(img):
    
    vectorLength=np.size(img)
    img1=np.reshape(img,vectorLength)
    return img1

def reshape_VectorTo3D(img, shape=[5,92,92]):
    
    img2=np.reshape(img, shape)
    return img2




#####################################
#      Training and test Set
#####################################
class dataset_(object):
    
    def __init__(self, num_patient_tr, data_dir_training, num_patient_tst, data_dir_test):
        self.data_dir_training = data_dir_training
        self.data_dir_test = data_dir_test
        self.num_patient_tr = num_patient_tr
        self.num_patient_tst = num_patient_tst


    def create_training_validating_data_abtImage(self): # abt channels
        
        training=[]
        label=[]
        patients_training=os.listdir(self.data_dir_training)
        patients_training.sort()
        
        for u, patient in enumerate(tqdm(patients_training[0:self.num_patient_tr])):
            
            #print(patient)
            folder = patients_training[u]
            path_or = os.path.join(self.data_dir_training, folder)
            path_seg = os.path.join(self.data_dir_training, folder)
            
            # original images ADC
            files_or = os.listdir(path_or)
            #print(files_or)       
            files_seg = os.listdir(path_seg)
    
            name_seg_o1 = [fn for fn in files_or if 'cMap1'  in fn] 
            name_seg_o2 = [fn for fn in files_or if 'mag1'  in fn] 
            name_seg_o3 = [fn for fn in files_or if 'phi1'  in fn] 
            
            name_seg = [fn for fn in files_or if not 'ROI'  in fn] 
            name_seg_WP = [fn for fn in name_seg if 'WP'  in fn] 
            name_seg_TZ = [fn for fn in name_seg if 'TZ'  in fn] 
    
            cMap1=sitk.ReadImage(os.path.join(path_or, name_seg_o1[0])) 
            cMap1=sitk.GetArrayFromImage(cMap1)
            smin=0.0; smax=1.0
            normalized_cMap1 = ( cMap1 - cMap1.min() ) * smax / ( cMap1.max() - cMap1.min() ) + smin
    
            mag1=sitk.ReadImage(os.path.join(path_or, name_seg_o2[0])) 
            mag1=sitk.GetArrayFromImage(mag1)
            smin=0.0; smax=1.0
            normalized_mag1 = ( mag1 - mag1.min() ) * smax / ( mag1.max() - mag1.min() ) + smin
            
            phi1=sitk.ReadImage(os.path.join(path_or, name_seg_o3[0])) 
            phi1=sitk.GetArrayFromImage(phi1)
            smin=0.0; smax=1.0
            normalized_phi1 = ( phi1 - phi1.min() ) * smax / ( phi1.max() - phi1.min() ) + smin
        
            seg_WP=sitk.ReadImage(os.path.join(path_seg, name_seg_WP[0])) 
            seg_WP=sitk.GetArrayFromImage(seg_WP)
            smin=0.0; smax=1.0
            normalized_WP_seg = ( seg_WP - seg_WP.min() ) * smax / ( seg_WP.max() - seg_WP.min() ) + smin
    
            seg_TZ=sitk.ReadImage(os.path.join(path_seg, name_seg_TZ[0])) 
            seg_TZ=sitk.GetArrayFromImage(seg_TZ)
            smin=0.0; smax=1.0
            normalized_TZ_seg = ( seg_TZ - seg_TZ.min() ) * smax / ( seg_TZ.max() - seg_TZ.min() ) + smin
            
            size = 128
            
            abImage_or = np.zeros((np.shape(normalized_cMap1)[0],size, size, 3), 'float32')
            abImage_seg = np.zeros((np.shape(normalized_cMap1)[0],size, size, 4), 'float32')
            
            
            for ii in range(np.shape(normalized_cMap1)[0]):
       
                abImage_or[ii,:,:,0] = normalized_cMap1[ii, :, :]
                abImage_or[ii,:,:,1] = normalized_mag1[ii, :, :]
                abImage_or[ii,:,:,2] = normalized_phi1[ii, :, :]
                # masks
                abImage_seg[ii,:,:,0] = normalized_WP_seg[ii, :, :]
                neg = 1.0 - normalized_WP_seg[ii, :, :]
                abImage_seg[ii,:,:,1] = neg
    
                abImage_seg[ii,:,:,2] = normalized_TZ_seg[ii, :, :]
                neg = 1.0 - normalized_TZ_seg[ii, :, :]
                abImage_seg[ii,:,:,3] = neg
                
            for iii in range(np.shape(normalized_cMap1)[0]):
                training.append([abImage_or[iii,:,:,:], abImage_seg[iii,:,:,:]])
            
        return training
    
    #training_data = create_training_validating_data_abtImage(3, self.data_dir_training) 
  
    def create_testing_abtImage(self): # abt channels
        
        self.sizess = []
        training=[]
        label=[]
        patients_training=os.listdir(self.data_dir_test)
        patients_training.sort()
    
        for u, patient in enumerate(tqdm(patients_training[0:self.num_patient_tst])):
            
            print(patient)
            folder = patients_training[u]
            path_or = os.path.join(self.data_dir_test, folder)
            path_seg = os.path.join(self.data_dir_test, folder)
    
            files_or = os.listdir(path_or)

            files_seg = os.listdir(path_seg)
            name_seg_o1 = [fn for fn in files_or if 'cMap1'  in fn] 
            name_seg_o2 = [fn for fn in files_or if 'mag1'  in fn] 
            name_seg_o3 = [fn for fn in files_or if 'phi1'  in fn] 
            
            name_seg = [fn for fn in files_or if not 'ROI'  in fn] 
            name_seg_WP = [fn for fn in name_seg if 'WP'  in fn] 
            name_seg_TZ = [fn for fn in name_seg if 'TZ'  in fn] 

            cMap1=sitk.ReadImage(os.path.join(path_or, name_seg_o1[0])) 
            cMap1=sitk.GetArrayFromImage(cMap1)
            smin=0.0; smax=1.0
            normalized_cMap1 = ( cMap1 - cMap1.min() ) * smax / ( cMap1.max() - cMap1.min() ) + smin
    
            mag1=sitk.ReadImage(os.path.join(path_or, name_seg_o2[0])) 
            mag1=sitk.GetArrayFromImage(mag1)
            smin=0.0; smax=1.0
            normalized_mag1 = ( mag1 - mag1.min() ) * smax / ( mag1.max() - mag1.min() ) + smin
            
            phi1=sitk.ReadImage(os.path.join(path_or, name_seg_o3[0])) 
            phi1=sitk.GetArrayFromImage(phi1)
            smin=0.0; smax=1.0
            normalized_phi1 = ( phi1 - phi1.min() ) * smax / ( phi1.max() - phi1.min() ) + smin
            
            seg_WP=sitk.ReadImage(os.path.join(path_seg, name_seg_WP[0])) 
            seg_WP=sitk.GetArrayFromImage(seg_WP)
            smin=0.0; smax=1.0
            normalized_WP_seg = ( seg_WP - seg_WP.min() ) * smax / ( seg_WP.max() - seg_WP.min() ) + smin
    
            seg_TZ=sitk.ReadImage(os.path.join(path_seg, name_seg_TZ[0])) 
            seg_TZ=sitk.GetArrayFromImage(seg_TZ)
            smin=0.0; smax=1.0
            normalized_TZ_seg = ( seg_TZ - seg_TZ.min() ) * smax / ( seg_TZ.max() - seg_TZ.min() ) + smin
            
            size = 128
            
            abImage_or = np.zeros((np.shape(normalized_cMap1)[0],size, size, 3), 'float32')
            abImage_seg = np.zeros((np.shape(normalized_cMap1)[0],size, size, 4), 'float32')
                    
            for ii in range(np.shape(normalized_cMap1)[0]):
       
                abImage_or[ii,:,:,0] = normalized_cMap1[ii, :, :]
                abImage_or[ii,:,:,1] = normalized_mag1[ii, :, :]
                abImage_or[ii,:,:,2] = normalized_phi1[ii, :, :]
                
                abImage_seg[ii,:,:,0] = normalized_WP_seg[ii, :, :]
                neg = 1.0 - normalized_WP_seg[ii, :, :]
                abImage_seg[ii,:,:,1] = neg
    
                abImage_seg[ii,:,:,2] = normalized_TZ_seg[ii, :, :]
                neg = 1.0 - normalized_TZ_seg[ii, :, :]
                abImage_seg[ii,:,:,3] = neg
                            
            for iii in range(np.shape(normalized_cMap1)[0]):                        
                training.append([abImage_or[iii,:,:,:], abImage_seg[iii,:,:,:]])
                             
        return training
    


