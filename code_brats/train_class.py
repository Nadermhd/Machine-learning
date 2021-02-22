#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:08:02 2021

@author: nader
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import SimpleITK as sitk
import Models_3_out as M
import keras
import random
from keras.optimizers import Adam, SGD 
from save_load import save_models, load_models

#####################################
#      Training and test Class
#####################################

class train_valid_test():
    def __init__(self, dataset, n_epochs=10, batch_size=5, im=[], lb=[], model=[], precentage = 0.2, out_dir= ''):
        self.d = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.total_num = 10 #484
        train_num = int(np.floor( precentage * self.total_num))
        self.patients_train = train_num 
        self.patients_valid =  self.total_num - self.patients_train
        print('################### Prepare data ##################\n')
        print('Total number of patients {}, train : {}, validation: {}\n'
              .format(self.total_num, self.patients_train, self.patients_valid))
        
        self.count = 0
        self.count_v = 0
        self.patient_num = 0 # start index 
        self.patient_num_v = self.patients_train # start index 
        self.im = im
        self.lb = lb
        self.history_tr = []
        self.history_vl = []
        self.model = model
        self.out_dir = out_dir
        
    def train(self):
        print('############## Training loop, num of epochs: {}, num of patinets: {} ##############\n'
              .format(self.n_epochs, self.patients_train))
        
        for epoch in range(self.n_epochs):
            print('############## Training ##############')
            
            for p in range(self.patients_train):
                self.steps_train = self.im.shape[0] // self.batch_size
                self.steps_train = 5 # to be deleted
                
                for s in range(self.steps_train): # batches per patient: determined by slice_count // batch size
                    print('--Epoch {} out of {}, batch {} out of {}, pat {} out of {}'
                          .format(epoch, self.n_epochs, self.count, self.steps_train, self.patient_num, self.patients_train))
                    print('batch index: ', s * self.batch_size, (s * self.batch_size) + self.batch_size)
                    features = self.im[s * self.batch_size : (s * self.batch_size) + self.batch_size, ...]
                    features = features.astype('float32')
                    labels = self.lb[s * self.batch_size : (s * self.batch_size) + self.batch_size, ...]
                    labels = np.expand_dims(labels.astype('float32'), axis=-1)
                    # print('shape:', features.shape)
                    loss = self.model.train_on_batch(features, [labels, labels])
                    self.count += 1
                    self.history_tr.append(loss)
                    print('Total loss:', loss[0], 'Loss1: ', loss[1],'Loss2: ', loss[2], 'Dsc1: ', loss[3], 'Dsc2: ', loss[4])
                    
                    if self.count >= self.steps_train:
                        self.patient_num += 1
                        self.count = 0
                        self.im , self.lb, slices = self.d.create_training_validating_data_abtImage(self.patient_num)

            print('End of epoch .. re-read dataset, val loop')
            ######################
            self.history_vl = self.validation()
            ######################
            # resest index
            self.patient_num = 0
            self.im , self.lb, slices = self.d.create_training_validating_data_abtImage(self.patient_num)
            
            print('#####################################')
            print('#### Saveing model at epoch: ', epoch, '#####')
            print('#####################################')
            filename1 = self.out_dir + '/model_%03d_m.h5' % (epoch)
            filename2 = self.out_dir + '/model_%03d_w.h5' % (epoch)
            self.model.save(filename1)
            self.model.save_weights(filename2)
            
        return self.history_tr, self.history_vl
                      
    def validation(self):
        print('############## Validation loop, num of patients: {} ##############'.format(self.patients_valid))
        
        for pv in range(self.patients_valid):
            self.im , self.lb, slices = self.d.create_training_validating_data_abtImage(self.patient_num_v)
            self.steps_valid = self.im.shape[0] // self.batch_size
            self.steps_valid = 5 # to be deleted
            
            for v in range(self.steps_valid):
                print('Validation batch {} out of {} for pat {} out of {}, pat_idx {} out of total indx {}'
                          .format(v, self.steps_valid, pv, self.patients_valid, self.count_v, self.patient_num_v))
                
                features = self.im[v * self.batch_size : (v * self.batch_size) + self.batch_size, ...]
                features = features.astype('float32')
                
                labels = self.lb[v * self.batch_size : (v * self.batch_size) + self.batch_size, ...]
                labels = np.expand_dims(labels.astype('float32'), axis=-1)
                
                loss = self.model.test_on_batch(features, [labels, labels])
                self.count_v += 1
                self.history_vl.append(loss)
                
                print('Total loss:', loss[0], 'Loss1: ', loss[1],'Loss2: ', loss[2], 'Dsc1: ', loss[3], 'Dsc2: ', loss[4])
                
            self.patient_num_v += 1
            self.count_v = 0
            
        # reset the index    
        self.patient_num_v = self.patients_train
        
        return self.history_vl
    
    # to be modified 
    # def test(self):
    #     history_ts = []
    #     print('############## Test loop, num of patients: {} ##############'.format(self.patients_valid))
    #     for p in range(self.patients_valid):
    #         self.im , self.lb, slices = d.create_training_validating_data_abtImage(patient_num_v)
    #         self.steps_valid = self.im.shape[0] // self.batch_size
    #         self.steps_valid = 5 # to be deleted
            
    #         for v in range(self.steps_valid):
    #             print('Testing', v, 'out of: ', steps_valid, 'count_v', self.count_v, 'pat', self.patient_num_v)
                
    #             features = im[v * batch_size : (v * batch_size) + batch_size, ...]
    #             features = features.astype('float32')
                
    #             labels = lb[v * batch_size : (v * batch_size) + batch_size, ...]
    #             labels = np.expand_dims(labels.astype('float32'), axis=-1)
                
    #             loss = self.model.test_on_batch(features, [labels, labels])
    #             self.count_v += 1
    #             history_ts.append(loss)
                
    #             print('Total loss:', loss[0], 'Loss1: ', loss[1],'Loss2: ', loss[2], 'Dsc1: ', loss[3], 'Dsc2: ', loss[4])
                
    #         self.patient_num_v += 1
    #         self.count_v = 0
    #     # reset the index    
    #     self.patient_num_v = self.patients_valid
        
    #     return history_ts

# n_epochs=10
# batch_size=5
# train_num = 10    
# precentage = 0.6 # train - valid       
# opt = SGD(lr=0.01, momentum=0.90, decay=1e-6)
# model = M.unet_normal(opt,input_size=(240, 240, 4))

# h_tr, h_vl = train_valid_test(n_epochs=10, batch_size=5, im=im, lb=lb, model=model, precentage = 0.2).train()   
