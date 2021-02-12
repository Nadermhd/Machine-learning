# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:27:07 2020

@author: aldojn
"""
# Load the TensorBoard notebook extension.
#%load_ext tensorboard
import numpy as np 
import os
import openslide as ops
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import models_ as M
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD 
import keras
from datetime import datetime
from tensorflow.python.keras.callbacks import TensorBoard
print(keras.__version__)

class PandaDataset():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_dir = dataset_dir + '\\train_images'
        self.masks_dir = dataset_dir + '\\train_lablels_masks'
        self.patch_size = 256
        self.batch_size = 1
        self.shuffle = False
        self.ct_tr = 0 # count the number of runs of a specific method 'train' [0 : 8616]
        self.ct_va = 0 # count the number of runs of a specific method      [8616 : 9616]
        self.ct_ts = 0 # count the number of runs of a specific method      [9616 : 10616]
        self.len_ = 0
        self.level_count = 2 # 0: 5300, 1: 385, 2: 20-40
        self.on_epoch_end()
        self._print = True
        #self.index = image_id #'00a7fb880dc12c5de82df39b30533da9'
    
    def on_epoch_end(self):
        tr, _ = self.read_csv()
        
        self.indexes = np.arange(len(tr))
        self.len_ = len(tr)
        
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        self.tr_adx = self.indexes[:-2000] # 80616
        self.va_adx = self.indexes[len(tr)-2000:len(tr)-1000] # 1000
        self.ts_adx = self.indexes[len(tr)-1000:len(tr)] # 1000
        
    def __getitem__(self, idx):
        
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]
        slide = ops.OpenSlide(self.image_dir + '\\{}.tiff'.format(self.indexes))
        mask = ops.OpenSlide(self.masks_dir + '\\{}_mask.tiff'.format(self.indexes))
        
        img = slide.get_thumbnailc(256, 256)
        return img
    
    def read_img(self, index):

        slide = ops.OpenSlide(self.image_dir + '\\{}.tiff'.format(self.index))
        mask = ops.OpenSlide(self.masks_dir + '\\{}_mask.tiff'.format(self.index))
        x,y = 0, 0
        slide_pil = ops.open_slide(self.image_dir + '\\{}.tiff'.format(self.index))
        i = slide_pil.read_region((x, y), self.level_count, slide_pil.level_dimensions[self.level_count])
        
        mask_pil = ops.open_slide(self.masks_dir + '\\{}_mask.tiff'.format(self.index))
        m = mask_pil.read_region((x, y), self.level_count, mask_pil.level_dimensions[self.level_count])
        mth = self.can_nocan_lbl(m)
        return i, slide, mth
        
    def read_img_tr(self, index):
        index = self.ct_tr
        self.name_slide = tr_df.iloc[self.tr_adx[index]]['image_id']
        gs = tr_df.iloc[index]['gleason_score']

        slide = ops.OpenSlide(self.image_dir + '\\{}.tiff'.format(self.name_slide))
        mask = ops.OpenSlide(self.masks_dir + '\\{}_mask.tiff'.format(self.name_slide))
        x,y = 0, 0
        slide_pil = ops.open_slide(self.image_dir + '\\{}.tiff'.format(self.name_slide))
        i = slide_pil.read_region((x, y), self.level_count, slide_pil.level_dimensions[self.level_count])
        
        mask_pil = ops.open_slide(self.masks_dir + '\\{}_mask.tiff'.format(self.name_slide))
        m = mask_pil.read_region((x, y), self.level_count, mask_pil.level_dimensions[self.level_count])
        if self._print:
            print('Read image: ', self.ct_tr, ', Out of:', self.tr_adx[index], 'name: ', self.name_slide, 'GS:', gs)
        self.ct_tr += 1
        if self.ct_tr >= 160: #len(self.tr_adx):
            self.ct_tr = 0
        self.data_prov = tr_df.iloc[self.tr_adx[index]]['data_provider']
        # threshold the mask 
        
        #mth = self.can_nocan_lbl(m)
        
        return i, m, gs
    
    def read_img_va(self, index):
        
        self.name_slide = tr_df.iloc[self.va_adx[index]]['image_id']
        gs = tr_df.iloc[index]['gleason_score']

        slide = ops.OpenSlide(self.image_dir + '\\{}.tiff'.format(self.name_slide))
        mask = ops.OpenSlide(self.masks_dir + '\\{}_mask.tiff'.format(self.name_slide))
        x,y = 0, 0
        slide_pil = ops.open_slide(self.image_dir + '\\{}.tiff'.format(self.name_slide))
        i = slide_pil.read_region((x, y), self.level_count, slide_pil.level_dimensions[self.level_count])
        
        mask_pil = ops.open_slide(self.masks_dir + '\\{}_mask.tiff'.format(self.name_slide))
        m = mask_pil.read_region((x, y), self.level_count, mask_pil.level_dimensions[self.level_count])
        if self._print:
            print('Read image_va: ', self.ct_va, ', Out of:', self.va_adx, 'name: ', self.name_slide, 'GS:', gs)
        self.ct_va += 1
        if self.ct_va >= 3: # len(self.va_adx):
            self.ct_va = 0
        mth = self.can_nocan_lbl(m)
        return i, mth, gs
    
    def read_img_ts(self, index):
        
        self.name_slide = tr_df.iloc[self.ts_adx[index]]['image_id']
        gs = tr_df.iloc[index]['gleason_score']

        slide = ops.OpenSlide(self.image_dir + '\\{}.tiff'.format(self.name_slide))
        mask = ops.OpenSlide(self.masks_dir + '\\{}_mask.tiff'.format(self.name_slide))
        x,y = 0, 0
        slide_pil = ops.open_slide(self.image_dir + '\\{}.tiff'.format(self.name_slide))
        i = slide_pil.read_region((x, y), self.level_count, slide_pil.level_dimensions[self.level_count])
        
        mask_pil = ops.open_slide(self.masks_dir + '\\{}_mask.tiff'.format(self.name_slide))
        m = mask_pil.read_region((x, y), self.level_count, mask_pil.level_dimensions[self.level_count])
        if self._print:
            print('Read image_va: ', self.ct_ts, ', Out of:', self.ts_adx, 'name: ', self.name_slide, 'GS:', gs)
        self.ct_ts += 1
        if self.ct_ts >= len(self.ts_adx):
            self.ct_ts = 0
        mth = self.can_nocan_lbl(m)
        return i, mth, gs
    
    def read_csv(self):
        train = pd.read_csv(os.path.join(self.dataset_dir, "train.csv"))
        test = pd.read_csv(os.path.join(self.dataset_dir, "test.csv"))
        
        # df_id = train.loc[train['image_id'] == self.index]
        # gs = df_id['gleason_score'].values[0]
        
        return train, test#, gs
    
    def get_tile(self, i, m):
        
        pad_i, pad_m = i, m # self.pad_img(i, m)
        h, w = pad_i.shape[0], pad_i.shape[1]
        tiles_i = []
        tiles_m = []
        if self._print:
            print('height: {}, width: {}'.format(h, w))
        
        for x in range(0, w, self.patch_size//2):
            for y in range(0, h, self.patch_size//2):
                tile_i = pad_i[x:x+self.patch_size//2, y:y+self.patch_size//2, :]
                tile_m = pad_m[x:x+self.patch_size//2, y:y+self.patch_size//2]
                # check if tile has relevant values
                if len(np.where(tile_m == 1)[0]) >= 10:
                    tiles_i.append(tile_i)
                    tiles_m.append(tile_m)
        # different num of tiles which depends on the content of original image
        self.noftiles = len(tiles_i)
        if self._print:
            print('Num of tiles:', self.noftiles)
        return tiles_i, tiles_m

    def pad_img(self, i, m):
        
        h, w = i.height, i.width
        pad_x = int(h- self.patch_size*np.ceil(h//self.patch_size) )
        pad_y = int(w- self.patch_size*np.ceil(w//self.patch_size) )
        img = np.pad(i,[[pad_x//2,pad_x//2],[pad_y//2,pad_y//2],[0,0]],
                         'constant', constant_values=(255))
        msk = np.pad(m,[[pad_x//2,pad_x//2],[pad_y//2,pad_y//2],[0,0]],
                         'constant', constant_values=(0))
        self.t_x = img.shape[0] // 256
        self.t_y = img.shape[1] // 256
        self.n_tiles = self.t_x * self.t_y
        mth = self.can_nocan_lbl(msk) 
        return img, mth 
    
    def plot_tile_mask(self, i, m):
        ii, im = i, m # self.get_tile(i, m)
        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
        for i in range(0,20, 2):
            plt.subplot(5, 4, i+1)
            plt.imshow(ii[i], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
            plt.subplot(5, 4, i+2)
            imm = im[i]
            plt.imshow(imm, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
        
    def plot_img_msk(self, i, m):
        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
        plt.subplot(1, 2, 1)
        plt.imshow(i, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
        plt.subplot(1, 2, 2)
        #mth = self.can_nocan_lbl(m)
        plt.imshow(m) #m[...,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
   
    def normalize(self, i, m):
        nrm_i = i.astype(np.float32) / 255 # 1.0 - (i/255.0)
        nrm_i = (i - np.mean(i))/np.std(i)
        return i, m
    
    def can_nocan_lbl(self, m):
        tr, _ = self.read_csv()
        
        # j = tr.iloc['karolinska']['data_provider']
        # gs = tr.iloc[j]['gleason_score']
        
        print(np.shape(m))
        if self.data_prov == 'karolinska': 
            print('th_k:', 1)
            mth = np.where(m[..., 0]>=2, 1, 0).astype(np.uint8)
        else:
            print('th_r:', 2)
            mth = np.where(m[..., 0]>=3, 1, 0).astype(np.uint8)
        return mth

def conc(m, lc):
    le = len(m)
    arr = np.empty((le, 128, 128, lc))
    for k in range(le):
        temp = np.reshape(m[k], [128, 128, lc])
        #print(temp.shape)
        arr[k, ...] = temp
    return arr
        
pand = PandaDataset('D:\Work\panda')
tr_df, ts_df = pand.read_csv()
# tiles amd masks as one batch 

dir_to_save = 'D:\\Work\\panda\\npy'
os.chdir(dir_to_save)


train = True

print('############################################################################')
print('#                             THE U-NET MODEL                              #')
print('############################################################################')

opt = SGD(lr=0.01, momentum=0.90, decay=1e-6)
model = M.unet_normal(opt,input_size=(128, 128, 1))
model.summary()



datagen = ImageDataGenerator(rescale=1. / 255)
seed = 1
train_it_ims = datagen.flow_from_directory( 'train\\images\\',class_mode=None,color_mode='grayscale', target_size=(128, 128), batch_size=35, seed = 1)
train_it_msk = datagen.flow_from_directory( 'train\\masks\\',class_mode=None,color_mode='grayscale', target_size=(128, 128), batch_size=35, seed = 1)
train_it = zip(train_it_ims, train_it_msk)

val_it_ims = datagen.flow_from_directory( 'val\\images\\',class_mode=None,color_mode='grayscale', target_size=(128, 128), batch_size=35, seed = 1)
val_it_msk = datagen.flow_from_directory( 'val\\masks\\',class_mode=None,color_mode='grayscale', target_size=(128, 128), batch_size=35, seed = 1)
val_it = zip(val_it_ims, val_it_msk)



x = train_it_ims.__getitem__(1)
y = train_it_msk.__getitem__(1)

#################################################
import tensorflow as tf
import skimage

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)

class TensorBoardImage(keras.callbacks.Callback):
    def __init__(self, tag):
        super().__init__() 
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img = skimage.data.astronaut()
        # Do something to the image
        img = (255 * skimage.util.random_noise(img)).astype('uint8')

        image = make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return

tbi_callback = TensorBoardImage('Image Example')
###################################################################
import keras.backend as K
from losses import dsc, tp, tn 

def pixel_difference(y_true, y_pred):
    '''
    Custom metrics for comparison of images
    pixel by pixel. 
    '''
    cof = 100/(128*128*35)
    return cof*K.sum(K.abs(y_true - y_pred))

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[pixel_difference, dsc, tp, tn])
#import tensorflow as tf

import keras
from PIL import Image
import io
import numpy as np
import tensorflow as t
class ImageHistory(keras.callbacks.Callback):
    ...
    def __init__(self, tensor_board_dir, data, last_step, draw_interval):
       super(ImageHistory, self).__init__()
       self.draw_interval = draw_interval
       self.last_step = last_step
       self.tensor_board_dir = tensor_board_dir
       self.data = data
  
    def on_batch_end(self, batch, logs={}):
        if batch % self.draw_interval == 0:
            images = []
            labels = []
            for item in self.data:
                image_data = item[0]
                label_data = item[1]
                y_pred = self.model.predict(image_data)
                images.append(y_pred)
                labels.append(label_data)
            image_data = np.concatenate(images,axis=2)
            label_data = np.concatenate(labels,axis=2)
            data = np.concatenate((image_data,label_data), axis=1)
            self.last_step += 1
            self.saveToTensorBoard(data, 'batch',
               self.last_step*self.draw_interval)
        return
    
    def make_image(self, npyfile):
        """
        Convert an numpy representation image to Image protobuf.
        taken and updated from 
        https://github.com/lanpa/tensorboard-pytorch/
        """
        height, width, channel = npyfile.shape
        image = Image.frombytes('L',(width,height),
                               npyfile.tobytes())
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.compat.v1.Summary.Image(height=height,
                             width=width, colorspace=channel,
                             encoded_image_string=image_string)
    
    def saveToTensorBoard(self, npyfile, tag, epoch):
        data = npyfile[0,:,:,:]
        image = (((data - data.min()) * 255) / 
             (data.max() - data.min())).astype(np.uint8)
        image = self.make_image(image)
        summary = tf.compat.v1.Summary(
             value=[tf.compat.v1.Summary.Value(tag=tag,
                 image=image)])
        writer = tf.compat.v1.summary.FileWriter(
                 self.tensor_board_dir)
        writer.add_summary(summary, epoch)
        writer.close()

logdir = "D:\\Work\\panda\\logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq = 0, write_graph = True, write_images = True)
last_step = 10

image_history = ImageHistory(tensor_board_dir=logdir,
        data=train_it, last_step=last_step, draw_interval=100)
##############################################
if train:
    mcp_save = ModelCheckpoint('weight_prostate', save_best_only=True, monitor='val_loss',mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit_generator(train_it,
     	epochs=100,
        steps_per_epoch=19,
     	shuffle=True,
     	verbose=1,
     	validation_data=val_it, validation_steps=1, callbacks=[mcp_save, reduce_lr_loss, tensorboard_callback])

# to evaluate
else: 
    its, mts, gs = pand.read_img_tr(1)
    pits, pmts = pand.pad_img(its, mts)
    iits, imts = pand.get_tile(pits, pmts)
    ots = np.reshape(iits, [len(iits), 128, 128, 4])
    pts = np.reshape(imts, [len(iits), 128, 128, 1])
                
    weights = 'D:\\Work\\panda' + '\\weight_patho' 
    
    model.load_weights(weights)
    predictions = model.predict(ots, batch_size=100, verbose=1)
    
    ii, im = ots, predictions # self.get_tile(i, m)
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    for i in range(0,20, 2):
        plt.subplot(5, 4, i+1)
        plt.imshow(ii[i, ...])#, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
        plt.subplot(5, 4, i+2)
        imm = im[i, ..., 0]
        plt.imshow(imm, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
    
  
    
    
    