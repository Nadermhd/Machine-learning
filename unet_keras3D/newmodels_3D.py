#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:20:55 2020

@author: nader
"""
import cv2
import time
import os
import h5py

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, \
    UpSampling3D, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import AveragePooling2D, average, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import glorot_normal, random_normal, random_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.applications import VGG19, densenet
from keras.models import load_model

import numpy as np
import tensorflow as tf
import losses
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # roc curve tools
from sklearn.model_selection import train_test_split

img_row = 64
img_col = 64
img_dpth = 32
img_chan = 3
epochnum = 50
batchnum = 16
smooth = 1.
input_size = (img_row, img_col, img_dpth, img_chan)
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
kinit = 'glorot_normal'

def UnetConv3D(input, outdim, is_batchnorm, name):
    x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name + '_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name + '_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x



def unet(opt, input_size, lossfxn):
    inputs = Input(shape=input_size)
    conv1 = UnetConv3D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = UnetConv3D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = UnetConv3D(pool2, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = UnetConv3D(pool3, 256, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv5)
    #print(conv5)
    up6 = concatenate(
        [Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4],
        axis=4)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv7)

    up8 = concatenate(
        [Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=kinit, padding='same')(conv7), conv2],
        axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up8)

    up9 = concatenate(
        [Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=kinit, padding='same')(conv8), conv1],
        axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)
    
    conv9 = Conv3D(6, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)

    out_1 =  Conv3D(2, (1, 1, 1), activation='softmax', name='final_1')(conv9)
    out_2 =  Conv3D(2, (1, 1, 1), activation='softmax', name='final_2')(conv9)
    out_3 =  Conv3D(2, (1, 1, 1), activation='softmax', name='final_3')(conv9)
    
    model = Model(inputs=[inputs], outputs=[out_1, out_2, out_3])
    
    loss = {'final_1': lossfxn,
            'final_2': lossfxn,
            'final_3': lossfxn}

    loss_weights = {'final_1': 1,
                    'final_2': 1,
                    'final_3': 1}
    
    model.compile(optimizer=opt, loss=loss, metrics=[losses.dsc, losses.tp, losses.tn], loss_weights=loss_weights)
    return model

# sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
# model = unet(sgd, input_size, losses.focal_tversky)
# model.summary()


def expend_as(tensor, rep, name):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep},
                       name='psi_up' + name)(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape, name):

    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding='same', name='xl' + name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv3D(inter_shape, (1, 1, 1), padding='same')(g)
    upsample_g = Conv3DTranspose(inter_shape, (3, 3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]),
                                 padding='same', name='g_up' + name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv3D(1, (1, 1, 1), padding='same', name='psi' + name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    #print(np.shape(sigmoid_xg), np.shape(x))
    #print(shape_theta_x[3], shape_g[3])
    upsample_psi = UpSampling3D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)  # 32
    #print(np.shape(upsample_psi), np.shape(x))
    upsample_psi = expend_as(upsample_psi, shape_x[4], name)

    y = multiply([upsample_psi, x], name='q_attn' + name)

    result = Conv3D(shape_x[4], (1, 1, 1), padding='same', name='q_attn_conv' + name)(y)
    result_bn = BatchNormalization(name='q_attn_bn' + name)(result)
    return result_bn

def UnetGatingSignal(input, is_batchnorm, name):
    ''' this is simply 1x1 convolution, bn, activation '''
    shape = K.int_shape(input)
    x = Conv3D(shape[3] * 1, (1, 1, 1), strides=(1, 1, 1), padding="same", kernel_initializer=kinit, name=name + '_conv')(
        input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_act')(x)
    return x


# plain old attention gates in u-net, NO multi-input, NO deep supervision
def attn_unet(opt, input_size, lossfxn):
    inputs = Input(shape=input_size)
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
    # conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
    # conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = UnetConv2D(pool4, 128, is_batchnorm=True, name='center')

    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                       kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate(
        [Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up1),
         attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up2),
         attn3], name='up3')

    up4 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up3),
         conv1], name='up4')
    out = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=kinit, name='final')(up4)

    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=opt, loss=lossfxn, metrics=[losses.dsc, losses.tp, losses.tn])
    return model


# regular attention unet with  deep supervision - exactly from paper (my intepretation)
def attn_reg_ds(opt, input_size, lossfxn):
    img_input = Input(shape=input_size, name='input_scale1')

    conv1 = UnetConv2D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
    # conv3 = Dropout(0.2,name='drop_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
    # conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = UnetConv2D(pool4, 512, is_batchnorm=True, name='center')

    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                       kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate(
        [Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up1),
         attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up2),
         attn3], name='up3')

    up4 = concatenate(
        [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up3),
         conv1], name='up4')

    conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv2D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])

    loss = {'pred1': lossfxn,
            'pred2': lossfxn,
            'pred3': lossfxn,
            'final': lossfxn}

    loss_weights = {'pred1': 1,
                    'pred2': 1,
                    'pred3': 1,
                    'final': 1}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
                  metrics=[losses.dsc])
    return model


# model proposed in my paper - improved attention u-net with multi-scale input pyramid and deep supervision

def attn_reg(opt, input_size, lossfxn):
    img_input = Input(shape=input_size, name='input_scale1')
    scale_img_2 = AveragePooling3D(pool_size=(2, 2, 2), name='input_scale2')(img_input)
    scale_img_3 = AveragePooling3D(pool_size=(2, 2, 2), name='input_scale3')(scale_img_2)
    scale_img_4 = AveragePooling3D(pool_size=(2, 2, 2), name='input_scale4')(scale_img_3)

    conv1 = UnetConv3D(img_input, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    input2 = Conv3D(64, (3, 3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    input2 = concatenate([input2, pool1], axis=4)
    conv2 = UnetConv3D(input2, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    input3 = Conv3D(128, (3, 3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = concatenate([input3, pool2], axis=4)
    conv3 = UnetConv3D(input3, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    input4 = Conv3D(256, (3, 3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = concatenate([input4, pool3], axis=4)
    conv4 = UnetConv3D(input4, 64, is_batchnorm=True, name='conv4')
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    center = UnetConv3D(pool4, 512, is_batchnorm=True, name='center')

    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu',
                                       kernel_initializer=kinit)(center), attn1], name='up1')

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate(
        [Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up1),
         attn2], name='up2')

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate(
        [Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up2),
         attn3], name='up3')

    up4 = concatenate(
        [Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up3),
         conv1], name='up4')

    conv6 = UnetConv3D(up1, 256, is_batchnorm=True, name='conv6')
    conv7 = UnetConv3D(up2, 128, is_batchnorm=True, name='conv7')
    conv8 = UnetConv3D(up3, 64, is_batchnorm=True, name='conv8')
    conv9 = UnetConv3D(up4, 32, is_batchnorm=True, name='conv9')

    out6 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred1')(conv6)
    out7 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred2')(conv7)
    out8 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred3')(conv8)
    out9 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])

    loss = {'pred1': lossfxn,
            'pred2': lossfxn,
            'pred3': lossfxn,
            'final': losses.tversky_loss}

    loss_weights = {'pred1': 1,
                    'pred2': 1,
                    'pred3': 1,
                    'final': 1}
    model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
                  metrics=[losses.dsc])
    print(np.shape(out6))
    print(np.shape(out7))
    print(np.shape(out8))
    print(np.shape(out9))
    return model
#sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
#model = attn_reg(sgd, input_size, losses.focal_tversky)
#model.summary()

n = 1