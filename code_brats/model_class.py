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

patients_training=os.listdir(data_dir_test)
patients_training.sort()


cost_plot=[]
loss_plot=[]
accu_plot=[]

IMG_SIZE = 64
SLICE_COUNT = 1

#####################################
#       Timer
#####################################

def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
    print('\nTic: Start timer..')

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")
        

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
# i = i[100]
# m = m[100]
# plt.subplot(2,1,1)
# plt.imshow(i[1, 50, ...])
# plt.subplot(2,1,2)
# plt.imshow(m[50, ...])

######################################
n_epochs = 10
batch_size = 5

total_num = len(im)
train_num = 10
valid_num = total_num - train_num

steps_train = train_num // batch_size
steps_valid = valid_num // batch_size

count = 0
count_v = 0
patient_num = 1
patient_num_v = valid_num
history_tr = []
history_vl = []

#####################################
#      Training and test Class
#####################################

class train_valid_test():
    def __init__(self, n_epochs=10, batch_size=5, im=[], lb=[], model=[], precentage = 0.2):
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
                    print('batch index: ', s * batch_size, (s * batch_size) + batch_size)
                    features = self.im[s * batch_size : (s * batch_size) + batch_size, ...]
                    features = features.astype('float32')
                    labels = self.lb[s * batch_size : (s * batch_size) + batch_size, ...]
                    labels = np.expand_dims(labels.astype('float32'), axis=-1)
                    
                    loss = self.model.train_on_batch(features, [labels, labels])
                    self.count += 1
                    self.history_tr.append(loss)
                    print('Total loss:', loss[0], 'Loss1: ', loss[1],'Loss2: ', loss[2], 'Dsc1: ', loss[3], 'Dsc2: ', loss[4])
                    
                    if self.count >= self.steps_train:
                        self.patient_num += 1
                        self.count = 0
                        self.im , self.lb, slices = d.create_training_validating_data_abtImage(self.patient_num)

            print('End of epoch .. re-read dataset, val loop')
            ######################
            self.history_vl = self.validation()
            ######################
            # resest index
            self.patient_num = 0
            self.im , self.lb, slices = d.create_training_validating_data_abtImage(self.patient_num)
            
        return self.history_tr, self.history_vl
                      
    def validation(self):
        print('############## Validation loop, num of patients: {} ##############'.format(self.patients_valid))
        for pv in range(self.patients_valid):
            self.im , self.lb, slices = d.create_training_validating_data_abtImage(patient_num_v)
            self.steps_valid = self.im.shape[0] // self.batch_size
            self.steps_valid = 5 # to be deleted
            
            for v in range(self.steps_valid):
                print('Validation batch {} out of {} for pat {} out of {}, pat_idx {} out of total indx {}'
                          .format(v, self.steps_valid, pv, self.patients_valid, self.count_v, self.patient_num_v))
                
                features = im[v * batch_size : (v * batch_size) + batch_size, ...]
                features = features.astype('float32')
                
                labels = lb[v * batch_size : (v * batch_size) + batch_size, ...]
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
    def test(self):
        history_ts = []
        print('############## Test loop, num of patients: {} ##############'.format(self.patients_valid))
        for p in range(self.patients_valid):
            self.im , self.lb, slices = d.create_training_validating_data_abtImage(patient_num_v)
            self.steps_valid = self.im.shape[0] // self.batch_size
            self.steps_valid = 5 # to be deleted
            
            for v in range(self.steps_valid):
                print('Testing', v, 'out of: ', steps_valid, 'count_v', self.count_v, 'pat', self.patient_num_v)
                
                features = im[v * batch_size : (v * batch_size) + batch_size, ...]
                features = features.astype('float32')
                
                labels = lb[v * batch_size : (v * batch_size) + batch_size, ...]
                labels = np.expand_dims(labels.astype('float32'), axis=-1)
                
                loss = self.model.test_on_batch(features, [labels, labels])
                self.count_v += 1
                history_ts.append(loss)
                
                print('Total loss:', loss[0], 'Loss1: ', loss[1],'Loss2: ', loss[2], 'Dsc1: ', loss[3], 'Dsc2: ', loss[4])
                
            self.patient_num_v += 1
            self.count_v = 0
        # reset the index    
        self.patient_num_v = self.patients_valid
        
        return history_ts

n_epochs=10
batch_size=5
train_num = 10    
precentage = 0.6 # train - valid       
opt = SGD(lr=0.01, momentum=0.90, decay=1e-6)
model = M.unet_normal(opt,input_size=(240, 240, 4))

# h_tr, h_vl = train_valid_test(n_epochs, 5, im, lb, model, 10).train() 
# im=[], lb=[], model=[], train_num = 10 
h_tr, h_vl = train_valid_test(n_epochs=10, batch_size=5, im=im, lb=lb, model=model, precentage = 0.2).train()   

# for epoch in range(n_epochs):
#     steps_train = 5 # delete when done
#     steps_valid = 5 # delete when done
    
#     for s in range(steps_train):
#         print(epoch, 'out of: ', n_epochs, s * batch_size, (s * batch_size) + batch_size, count, 'out of: ', steps_train, 'pat: ', patient_num)
#         features = im[s * batch_size : (s * batch_size) + batch_size, ...]
#         features = features.astype('float32')
        
#         labels = lb[s * batch_size : (s * batch_size) + batch_size, ...]
#         labels = np.expand_dims(labels.astype('float32'), axis=-1)
        
#         loss = model.train_on_batch(features, [labels, labels])
#         count += 1
#         history_tr.append(loss)
#         print('Total loss:', loss[0], 'Loss1: ', loss[1],'Loss2: ', loss[2], 'Dsc1: ', loss[3], 'Dsc2: ', loss[4])
         
#         if count >= steps_train:
            
#             print('End of epoch .. read new patient, val loop if needed')
#             ######################
#             for v in range(steps_valid):
#                 print('Validation', v, 'out of: ', steps_valid, 'count_v', count_v, 'pat', patient_num_v)
#                 im , lb, slices = d.create_training_validating_data_abtImage(patient_num_v)
#                 features = im[v * batch_size : (v * batch_size) + batch_size, ...]
#                 features = features.astype('float32')
                
#                 labels = lb[v * batch_size : (v * batch_size) + batch_size, ...]
#                 labels = np.expand_dims(labels.astype('float32'), axis=-1)
                
#                 loss = model.test_on_batch(features, [labels, labels])
#                 count += 1
#                 history_vl.append(loss)
#                 print('Total loss:', loss[0], 'Loss1: ', loss[1],'Loss2: ', loss[2], 'Dsc1: ', loss[3], 'Dsc2: ', loss[4])
                
#                 if count_v >= steps_valid:
#                     count = 0
#                     count_v = 0
                
#                 patient_num_v += 1
#                 if patient_num_v >= total_num:
#                     patient_num_v = valid_num
#                     break
#             ######################
#             im , lb, slices = d.create_training_validating_data_abtImage(patient_num)
#             patient_num += 1
        
#         if patient_num >= train_num:
#             patient_num = 0
#             break
        

    

######################################


from layers import *
import horovod.tensorflow as hvd
from tensorflow.keras.utils import Progbar

#from runtime.run import train, evaluate, predict
from arguments import PARSER, parse_args
from losses_ import partial_losses

class UNET(tf.keras.Model):
    def __init__(self):
        super(UNET, self).__init__()
        self.input_block = InputBlock(filters=64)
        self.down_blocks = [DownsampleBlock(filters, idx) for idx, filters in enumerate([512, 256, 128])]
        
        self.bottleneck = BottleneckBlock(1024)

        self.up_blocks = [UpsampleBlock(filters, idx) for idx, filters in enumerate([512, 256, 128])]
        self.output_block = OutputBlock(filters=64, n_classes=1)
        
    def call(self, x, training=True):
        skip_connections = []
        out, residual = self.input_block(x)
        skip_connections.append(residual)
        
        for down_block in self.down_blocks:
            out, residual = down_block(out)
            skip_connections.append(residual)
            
        out = self.bottleneck(out, training)
        
        for up_block in self.up_blocks:
            out = up_block(out, skip_connections.pop())
            
        out = self.output_block(out, skip_connections.pop())
        
        return out

model = UNET()
#opt = SGD(lr=0.01, momentum=0.90, decay=1e-6)
# model = M.unet_normal('Adam',input_size=(240, 240, 4))
# model.summary()
# metrics_names = ['acc','pr'] 

def main():  
    hvd.init()
    
    n_epochs = 10
    batch_size = 5
    step = len(im) // batch_size
    
    params = parse_args(PARSER.parse_args())
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    pb_i = Progbar(step, stateful_metrics=metrics_names)
    count = 0
    for epoch in range(n_epochs):
        
        if count >= step:
            count = 0
        
        features = im[epoch * batch_size : (epoch * batch_size) + batch_size]
        features = np.reshape(features, (len(features), features[0].shape[1], features[0].shape[2], features[0].shape[0]))
        features = features.astype('float32')
        
        labels = lb[epoch * batch_size : (epoch * batch_size) + batch_size]
        labels = np.reshape(labels, (len(labels), labels[0].shape[0], labels[0].shape[1], 1))
        labels = labels.astype('float32')
        print(features.shape, labels.shape)
        
        print('Epoch {} out of epochs {}'.format(epoch, n_epochs))
        
        
        
        for i, (features_, labels_) in enumerate(zip(features, labels)):
            
            with tf.GradientTape() as tape:
                
                output_map = model(features)
                
                crossentropy_loss, dice_loss = partial_losses(output_map, labels)
                added_losses = tf.add(crossentropy_loss, dice_loss, name='total_loss_ref')
                
                values=[('Xent', crossentropy_loss), ('added_losses', added_losses)]
                
                pb_i.add(1, values=values)
            
            # calculate the gradients using our tape and then update the
        	# model weights
            tape = hvd.DistributedGradientTape(tape)
            gradients = tape.gradient(added_losses, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Calculate something wrong here
            # val_total_loss = 0
            # val_total_acc = 0
            # total_val_num = 0
            # for bIdx, (val_X, val_y) in enumerate(val_batch):
            #     if bIdx >= features.shape[0]:
            #         break
            #     y_pred = model(val_X, training=False)
                
        print('Xen: ', crossentropy_loss, dice_loss, added_losses)
    
    
if __name__ == '__main__':
    main()
