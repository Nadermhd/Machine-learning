# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:19:00 2019

@author: aldojn
"""
from keras.models import Model
from keras.layers import Input, concatenate, Concatenate, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Reshape, core, Dropout, BatchNormalization, Conv2DTranspose, Activation, ConvLSTM2D
from keras.optimizers import Adam, SGD 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K 
from keras.utils.vis_utils import plot_model as plot
from random import shuffle
import numpy as np
from losses import dsc, tp, tn 
from keras.losses import binary_crossentropy
from keras.regularizers import l2



def My_conv_lstm_model(frames, channels, pixel_x, pixel_y, categories):

	trailer_input = Input(shape=(frames, channels, pixel_x, pixel_y), name='trailer_input')
	
	first_convlstm = ConvLSTM2D(filters=20, kernel_size=(3, 3)
		, data_format='channels_first'
		, recurrent_activation='hard_sigmoid'
		, activation='tanh'
		, padding='same', return_sequences=True)(trailer_input)
	first_batchNormalization = BatchNormalization()(first_convlstm)
	first_pooling = MaxPoolong3D(pool_size=(1, 2, 2), padding='same',
		data_format='channels_first')(first_batchNormalization)

	second_convlstm = ConvLSTM2D(filters=10, kernel_size=(3, 3)
		, data_format='channels_first'
		, padding='same', return_sequences=True)(first_pooling)
	second_batchNormalization = BatchNormalization()(second_convlstm)
	second_pooling = MaxPoolong3D(pool_size=(1, 3, 3), padding='same',
		data_format='channels_first')(second_batchNormalization)

	outputs = ''

	seq = Model(inputs=trailer_input, outputs=outputs, name='Model')

	return seq

def generate_npy():
	'''save each patient volume as a seperate npy file in a dataset folder'''
	pass

def generate_arrays(available_ids):
	 while True:
	 	shuffle(available_ids)
	 	for i in available_ids:
	 		img_3d = np.load('dataset/img_3d_{}'.format(i))
	 		lbl_3d = np.load('dataset/lbl_3d_{}'.format(i))
	 		yield (img_3d, lbl_3d)
             
def conv_factory(x, concat_axis, nb_filters, dropout_rate=None, weight_decay=1e-4):
	x = Conv2D(nb_filters, (3, 3),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
	x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)

	if dropout_rate:
        
	 	x = Dropout(dropout_rate)(x)

	return x

def transition(x, concat_axis, nb_filter,
               dropout_rate=None, weight_decay=1E-4, pool=True):
	x = Conv2D(nb_filter, (1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
	x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
	x = Activation('relu')(x)
	if dropout_rate:
	 	x = Dropout(dropout_rate)(x)

	xp = AveragePooling2D((2, 2), strides=(2, 2))(x)
	

	return x, xp

def denseblock(x, concat_axis, nb_layers, nb_filter, grwoth_rate, dropout_rate=None, weight_decay=1e-4):
	list_layers = [x]

	for i in range(nb_layers):
		x = conv_factory(x, concat_axis, grwoth_rate, dropout_rate, weight_decay)
		list_layers.append(x)
		x = Concatenate(axis=concat_axis)(list_layers)
		nb_filter += grwoth_rate
		return x, nb_filter
######################################################################
def BDLSTM_DenseUnet(opt, input_size=(256, 256, 1)):
	N = input_size[0]
	inputs = Input(input_size)

	conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1, nb_filter = denseblock(conv1, 3, 4,
                                  32, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv1, pool1 = transition(conv1, 3, 32, dropout_rate=None,
                       weight_decay=1E-4)

	conv2, nb_filter = denseblock(pool1, 3, 4,
                                  64, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv2, pool2 = transition(conv2, 3, 64, dropout_rate=None,
                       weight_decay=1E-4)

	conv3, nb_filter = denseblock(pool2, 3, 4,
                                  128, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv3, pool3 = transition(conv3, 3, 128, dropout_rate=None,
                       weight_decay=1E-4)
    
	conv4, nb_filter = denseblock(pool3, 3, 4,
                                  256, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv4, pool4 = transition(conv4, 3, 256, dropout_rate=None,
                       weight_decay=1E-4)

	conv5, nb_filter = denseblock(pool4, 3, 4,
                                  512, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv5, pool5 = transition(conv5, 3, 512, dropout_rate=None,
                       weight_decay=1E-4, pool=True)
    
    # bottelneck
	conv6, nb_filter = denseblock(pool5, 3, 4,
                                  512, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv6, pool6 = transition(conv6, 3, 512, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    
    # up stream
	up6 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
	up6 = BatchNormalization(axis=3)(up6)
	up6 = Activation('relu')(up6)
	
	x1 = Reshape(target_shape=(1, np.int32(N/16), np.int32(N/16), 512))(conv5)
	x2 = Reshape(target_shape=(1, np.int32(N/16), np.int32(N/16), 512))(up6)
	merge6  = concatenate([x1,x2], axis = 1) 
	merge6 = ConvLSTM2D(filters = 256, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
	conv7, nb_filter = denseblock(merge6, 3, 4,
                                  256, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv7, pool7 = transition(conv7, 3, 256, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up5 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
	up5 = BatchNormalization(axis=3)(up5)
	up5 = Activation('relu')(up5)
	
	x1 = Reshape(target_shape=(1, np.int32(N/8), np.int32(N/8), 256))(conv4)
	x2 = Reshape(target_shape=(1, np.int32(N/8), np.int32(N/8), 256))(up5)
	merge7  = concatenate([x1,x2], axis = 1) 
	merge7 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
            
	conv8, nb_filter = denseblock(merge7, 3, 4,
                                  128, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv8, pool8 = transition(conv8, 3, 128, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up4 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv8)
	up4 = BatchNormalization(axis=3)(up4)
	up4 = Activation('relu')(up4)
	
	x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 128))(conv3)
	x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 128))(up4)
	merge8  = concatenate([x1,x2], axis = 1) 
	merge8 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)
            
	conv9, nb_filter = denseblock(merge8, 3, 4,
                                  64, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv9, pool9 = transition(conv9, 3, 64, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up3 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv9)
	up3 = BatchNormalization(axis=3)(up3)
	up3 = Activation('relu')(up3)
	
	x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 64))(conv2)
	x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 64))(up3)
	merge9  = concatenate([x1,x2], axis = 1) 
	merge9 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge9)
            
	conv10, nb_filter = denseblock(merge9, 3, 4,
                                  32, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv10, pool10 = transition(conv10, 3, 32, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up2 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv10)
	up2 = BatchNormalization(axis=3)(up2)
	up2 = Activation('relu')(up2)
	
	x1 = Reshape(target_shape=(1, np.int32(N), np.int32(N), 32))(conv1)
	x2 = Reshape(target_shape=(1, np.int32(N), np.int32(N), 32))(up2)
	merge10  = concatenate([x1,x2], axis = 1) 
	merge10 = ConvLSTM2D(filters = 16, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge10)
            
	conv11, nb_filter = denseblock(merge10, 3, 4,
                                  16, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv11, pool11 = transition(conv11, 3, 16, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	out_1 =  Conv2D(2, (1, 1), activation='softmax', name='final_1')(conv11)
	out_2 =  Conv2D(2, (1, 1), activation='softmax', name='final_2')(conv11)
    
	model = Model(inputs=[inputs], outputs=[out_1, out_2])
    
	loss = {'final_1': binary_crossentropy,
            'final_2': binary_crossentropy}

	loss_weights = {'final_1': 1,
                    'final_2': 1}

	model.compile(optimizer=opt, loss=loss, metrics=[dsc, tp, tn], loss_weights=loss_weights)

	return model


def unet_3_out(opt, input_size = (256,256,1)):
    n_f = 32
    inputs = Input(input_size)
    conv1 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up6 = Conv2D(n_f*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
    merge6 = concatenate([drop3,up6], axis = 3)
    conv6 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv2,up7], axis = 3)
    conv7 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(n_f, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis = 3)
    conv8 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    ######
    out_1 =  Conv2D(2, (1, 1), activation='softmax', name='final_1')(conv8)
    out_2 =  Conv2D(2, (1, 1), activation='softmax', name='final_2')(conv8)
    out_3 =  Conv2D(2, (1, 1), activation='softmax', name='final_3')(conv8)
    model = Model(inputs=[inputs], outputs=[out_1, out_2, out_3])
    
    loss = {'final_1': binary_crossentropy,
            'final_2': binary_crossentropy,
            'final_3': binary_crossentropy}

    loss_weights = {'final_1': 1,
                    'final_2': 1,
                    'final_3': 1}

    model.compile(optimizer=opt, loss=loss, metrics=[dsc, tp, tn], loss_weights=loss_weights)
    ######

    return model

def unet_normal(opt, input_size = (256,256,1)):
    n_f = 32
    inputs = Input(input_size)
    conv1 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    
    up6 = Conv2D(n_f*16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(n_f*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(n_f*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(n_f*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9= concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    ######
    out_1 =  Conv2D(1, (1, 1), activation='sigmoid', name='final_1')(conv9)
    out_2 =  Conv2D(1, (1, 1), activation='sigmoid', name='final_2')(conv9)

    model = Model(inputs=[inputs], outputs=[out_1, out_2])

    loss = {'final_1': binary_crossentropy,
            'final_2': binary_crossentropy}

    loss_weights = {'final_1': 1,
                    'final_2': 1}

    model.compile(optimizer=opt, loss=loss, metrics=[dsc, tp, tn], loss_weights=loss_weights)
    ######

    return model

def DenseUnet(opt, input_size=(256, 256, 3)):
	N = input_size[0]
	inputs = Input(input_size)

	conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1, nb_filter = denseblock(conv1, 3, 4,
                                  32, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv1, pool1 = transition(conv1, 3, 32, dropout_rate=None,
                       weight_decay=1E-4)

	conv2, nb_filter = denseblock(pool1, 3, 4,
                                  64, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv2, pool2 = transition(conv2, 3, 64, dropout_rate=None,
                       weight_decay=1E-4)

	conv3, nb_filter = denseblock(pool2, 3, 4,
                                  128, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv3, pool3 = transition(conv3, 3, 128, dropout_rate=None,
                       weight_decay=1E-4)
    
	conv4, nb_filter = denseblock(pool3, 3, 4,
                                  256, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv4, pool4 = transition(conv4, 3, 256, dropout_rate=None,
                       weight_decay=1E-4)

	conv5, nb_filter = denseblock(pool4, 3, 4,
                                  512, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv5, pool5 = transition(conv5, 3, 512, dropout_rate=None,
                       weight_decay=1E-4, pool=True)
    
    # bottelneck
	conv6, nb_filter = denseblock(pool5, 3, 4,
                                  512, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv6, pool6 = transition(conv6, 3, 512, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    
    # up stream
	up6 = Conv2DTranspose(512, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
	up6 = BatchNormalization(axis=3)(up6)
	up6 = Activation('relu')(up6)
	
	x1 = Reshape(target_shape=(np.int32(N/16), np.int32(N/16), 512))(conv5)
	x2 = Reshape(target_shape=(np.int32(N/16), np.int32(N/16), 512))(up6)
	merge6  = concatenate([x1,x2], axis = 3) 
	#merge6 = ConvLSTM2D(filters = 256, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
	print(x1.shape, x2.shape, merge6.shape)       
	conv7, nb_filter = denseblock(merge6, 3, 4,
                                  256, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv7, pool7 = transition(conv7, 3, 256, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up5 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
	up5 = BatchNormalization(axis=3)(up5)
	up5 = Activation('relu')(up5)
	
	x1 = Reshape(target_shape=(np.int32(N/8), np.int32(N/8), 256))(conv4)
	x2 = Reshape(target_shape=(np.int32(N/8), np.int32(N/8), 256))(up5)
	merge7  = concatenate([x1,x2], axis = 3) 
	#merge7 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
            
	conv8, nb_filter = denseblock(merge7, 3, 4,
                                  128, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv8, pool8 = transition(conv8, 3, 128, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up4 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv8)
	up4 = BatchNormalization(axis=3)(up4)
	up4 = Activation('relu')(up4)
	
	x1 = Reshape(target_shape=(np.int32(N/4), np.int32(N/4), 128))(conv3)
	x2 = Reshape(target_shape=(np.int32(N/4), np.int32(N/4), 128))(up4)
	merge8  = concatenate([x1,x2], axis = 3) 
	#merge8 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)
            
	conv9, nb_filter = denseblock(merge8, 3, 4,
                                  64, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv9, pool9 = transition(conv9, 3, 64, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up3 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv9)
	up3 = BatchNormalization(axis=3)(up3)
	up3 = Activation('relu')(up3)
	
	x1 = Reshape(target_shape=(np.int32(N/2), np.int32(N/2), 64))(conv2)
	x2 = Reshape(target_shape=(np.int32(N/2), np.int32(N/2), 64))(up3)
	merge9  = concatenate([x1,x2], axis = 3) 
	#merge9 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge9)
            
	conv10, nb_filter = denseblock(merge9, 3, 4,
                                  32, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv10, pool10 = transition(conv10, 3, 32, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up2 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv10)
	up2 = BatchNormalization(axis=3)(up2)
	up2 = Activation('relu')(up2)
	
	x1 = Reshape(target_shape=(np.int32(N), np.int32(N), 32))(conv1)
	x2 = Reshape(target_shape=(np.int32(N), np.int32(N), 32))(up2)
	merge10  = concatenate([x1,x2], axis = 3) 
	#merge10 = ConvLSTM2D(filters = 16, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge10)
            
	conv11, nb_filter = denseblock(merge10, 3, 4,
                                  16, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv11, pool11 = transition(conv11, 3, 16, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	out_1 =  Conv2D(2, (1, 1), activation='softmax', name='final_1')(conv11)
	out_2 =  Conv2D(2, (1, 1), activation='softmax', name='final_2')(conv11)
    
	model = Model(inputs=[inputs], outputs=[out_1, out_2])
    
	loss = {'final_1': binary_crossentropy,
            'final_2': binary_crossentropy}

	loss_weights = {'final_1': 1,
                    'final_2': 1}

	model.compile(optimizer=opt, loss=loss,  loss_weights=loss_weights)

	return model

def unet_3_out_extend(opt, input_size = (256,256,1)):
    n_f = 16
    inputs = Input(input_size)
    conv1 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    ####
    conv5 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv55 = Conv2D(n_f*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv55 = Conv2D(n_f*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv55)
    drop55 = Dropout(0.5)(conv55)
    pool55 = MaxPooling2D(pool_size=(2, 2))(conv55)
    
    conv555 = Conv2D(n_f*64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool55)
    conv555 = Conv2D(n_f*64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv555)
    drop555 = Dropout(0.5)(conv555)
    ####
    

    up6 = Conv2D(n_f*32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop555))
    merge6 = concatenate([drop55,up6], axis = 3)
    conv6 = Conv2D(n_f*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(n_f*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(n_f*16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([drop5,up7], axis = 3)
    conv7 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(n_f*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([drop4,up8], axis = 3)
    conv8 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    ######
    up9 = Conv2D(n_f*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([drop3,up9], axis = 3)
    conv9 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up10 = Conv2D(n_f*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    merge10 = concatenate([conv2,up10], axis = 3)
    conv10 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    
    up11 = Conv2D(n_f*1, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv10))
    merge11 = concatenate([conv1,up11], axis = 3)
    conv11 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11)
    conv11 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    #######
    out_1 =  Conv2D(2, (1, 1), activation='softmax', name='final_1')(conv11)
    out_2 =  Conv2D(2, (1, 1), activation='softmax', name='final_2')(conv11)

    model = Model(inputs=[inputs], outputs=[out_1, out_2])

    loss = {'final_1': binary_crossentropy,
            'final_2': binary_crossentropy}

    loss_weights = {'final_1': 1,
                    'final_2': 1}

    model.compile(optimizer=opt, loss=loss, metrics=[dsc, tp, tn], loss_weights=loss_weights)
    ######

    return model

def DenseUnet1(opt, input_size=(256, 256, 1)):
	N = input_size[0]
	inputs = Input(input_size)
	n_f = 16

	conv1 = Conv2D(n_f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

	conv1, nb_filter = denseblock(conv1, 3, 4,
                                  n_f, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv1, pool1 = transition(conv1, 3, n_f, dropout_rate=None,
                       weight_decay=1E-4)
    #################################################
	conv2, nb_filter = denseblock(pool1, 3, 4,
                                  n_f*2, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv2, pool2 = transition(conv2, 3, n_f*2, dropout_rate=None,
                       weight_decay=1E-4)
    #################################################
	conv3, nb_filter = denseblock(pool2, 3, 4,
                                  n_f*4, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv3, pool3 = transition(conv3, 3, n_f*4, dropout_rate=None,
                       weight_decay=1E-4)
    #################################################    
	conv4, nb_filter = denseblock(pool3, 3, 4,
                                  n_f*8, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv4, pool4 = transition(conv4, 3, n_f*8, dropout_rate=None,
                       weight_decay=1E-4)
    #################################################
	conv5, nb_filter = denseblock(pool4, 3, 4,
                                  n_f*16, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv5, pool5 = transition(conv5, 3, n_f*16, dropout_rate=None,
                       weight_decay=1E-4, pool=True)
    
    ##################################################
	conv6, nb_filter = denseblock(pool5, 3, 4,
                                  n_f*32, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv6, pool6 = transition(conv6, 3, n_f*32, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    # bottelneck #################################################################################################

	up6 = Conv2DTranspose(n_f*16, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
	up6 = BatchNormalization(axis=3)(up6)
	up6 = Activation('relu')(up6)
	print(conv6, up6)
	x1 = Reshape(target_shape=(np.int32(N/16), np.int32(N/16), n_f*16))(conv5)
	x2 = Reshape(target_shape=(np.int32(N/16), np.int32(N/16), n_f*16))(up6)
	merge6  = concatenate([x1,x2], axis = 3) 
	#merge6 = ConvLSTM2D(filters = 256, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
   
	conv7, nb_filter = denseblock(merge6, 3, 4,
                                  n_f*8, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv7, pool7 = transition(conv7, 3, n_f*8, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
	
    ########################################################
	up5 = Conv2DTranspose(n_f*8, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
	up5 = BatchNormalization(axis=3)(up5)
	up5 = Activation('relu')(up5)
	print(conv4.shape)
	print(up5.shape)	
	x1 = Reshape(target_shape=(np.int32(N/8), np.int32(N/8), n_f*8))(conv4)
	x2 = Reshape(target_shape=(np.int32(N/8), np.int32(N/8), n_f*8))(up5)
	merge7  = concatenate([x1,x2], axis = 3) 
      
	conv8, nb_filter = denseblock(merge7, 3, 4,
                                  n_f*4, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv8, pool8 = transition(conv8, 3, n_f*4, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up4 = Conv2DTranspose(n_f*4, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv8)
	up4 = BatchNormalization(axis=3)(up4)
	up4 = Activation('relu')(up4)
	
	x1 = Reshape(target_shape=(np.int32(N/4), np.int32(N/4), n_f*4))(conv3)
	x2 = Reshape(target_shape=(np.int32(N/4), np.int32(N/4), n_f*4))(up4)
	merge8  = concatenate([x1,x2], axis = 3) 
	#merge8 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)
            
	conv9, nb_filter = denseblock(merge8, 3, 4,
                                  n_f*2, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv9, pool9 = transition(conv9, 3, n_f*2, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up3 = Conv2DTranspose(n_f*2, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv9)
	up3 = BatchNormalization(axis=3)(up3)
	up3 = Activation('relu')(up3)
	
	x1 = Reshape(target_shape=(np.int32(N/2), np.int32(N/2), n_f*2))(conv2)
	x2 = Reshape(target_shape=(np.int32(N/2), np.int32(N/2), n_f*2))(up3)
	merge9  = concatenate([x1,x2], axis = 3) 
	#merge9 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge9)
            
	conv10, nb_filter = denseblock(merge9, 3, 4,
                                  n_f*1, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv10, pool10 = transition(conv10, 3, n_f*1, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up2 = Conv2DTranspose(n_f*1, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv10)
	up2 = BatchNormalization(axis=3)(up2)
	up2 = Activation('relu')(up2)
	
	x1 = Reshape(target_shape=(np.int32(N/1), np.int32(N/1), n_f*1))(conv1)
	x2 = Reshape(target_shape=(np.int32(N/1), np.int32(N/1), n_f*1))(up2)
	merge10  = concatenate([x1,x2], axis = 3) 
	#merge10 = ConvLSTM2D(filters = 16, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge10)
            
	conv11, nb_filter = denseblock(merge10, 3, 4,
                                  n_f*1, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv11, pool11 = transition(conv11, 3, n_f*1, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	out_1 =  Conv2D(2, (1, 1), activation='softmax', name='final_1')(conv11)
	out_2 =  Conv2D(2, (1, 1), activation='softmax', name='final_2')(conv11)
	out_3 =  Conv2D(2, (1, 1), activation='softmax', name='final_3')(conv11)

	model = Model(inputs=[inputs], outputs=[out_1, out_2, out_3])

	loss = {'final_1': binary_crossentropy,
            'final_2': binary_crossentropy,
            'final_3': binary_crossentropy}

	loss_weights = {'final_1': 1,
                    'final_2': 1,
                    'final_3': 1}

	model.compile(optimizer=opt, loss=loss, metrics=[dsc, tp, tn], loss_weights=loss_weights)
	return model
#opt = SGD(lr=0.01, momentum=0.90, decay=1e-6)
#model = DenseUnet1(opt, input_size=(256, 256, 3))
#model.summary()
########################################################################################
def DenseUnet2(opt, input_size=(256, 256, 1)):
	N = input_size[0]
	inputs = Input(input_size)
	n_f = 16
	conv1 = Conv2D(n_f, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

	conv1, nb_filter = denseblock(conv1, 3, 4,
                                  n_f, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv1, pool1 = transition(conv1, 3, n_f, dropout_rate=None,
                       weight_decay=1E-4)
	conv1, nb_filter = denseblock(conv1, 3, 4,
                                  n_f, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv1, pool1 = transition(conv1, 3, n_f, dropout_rate=None,
                       weight_decay=1E-4)    
#######################
	conv2, nb_filter = denseblock(pool1, 3, 4,
                                  n_f*2, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv2, pool2 = transition(conv2, 3, n_f*2, dropout_rate=None,
                       weight_decay=1E-4)
	conv2, nb_filter = denseblock(conv2, 3, 4,
                                  n_f*2, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv2, pool2 = transition(conv2, 3, n_f*2, dropout_rate=None,
                       weight_decay=1E-4)
#######################    
	conv3, nb_filter = denseblock(pool2, 3, 4,
                                  n_f*4, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv3, pool3 = transition(conv3, 3, n_f*4, dropout_rate=None,
                       weight_decay=1E-4)
	conv3, nb_filter = denseblock(conv3, 3, 4,
                                  n_f*4, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv3, pool3 = transition(conv3, 3, n_f*4, dropout_rate=None,
                       weight_decay=1E-4)
#######################    
	conv4, nb_filter = denseblock(pool3, 3, 4,
                                  n_f*8, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv4, pool4 = transition(conv4, 3, n_f*8, dropout_rate=None,
                       weight_decay=1E-4)
	conv4, nb_filter = denseblock(conv4, 3, 4,
                                  n_f*8, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv4, pool4 = transition(conv4, 3, n_f*8, dropout_rate=None,
                       weight_decay=1E-4)
#######################    
	conv5, nb_filter = denseblock(pool4, 3, 4,
                                  n_f*16, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv5, pool5 = transition(conv5, 3, n_f*16, dropout_rate=None,
                       weight_decay=1E-4, pool=True)
	conv5, nb_filter = denseblock(conv5, 3, 4,
                                  n_f*16, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv5, pool5 = transition(conv5, 3, n_f*16, dropout_rate=None,
                       weight_decay=1E-4, pool=True)    
    # bottelneck #################################################################################################
	conv6, nb_filter = denseblock(pool5, 3, 4,
                                  n_f*32, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv6, pool6 = transition(conv6, 3, n_f*32, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
	conv6, nb_filter = denseblock(conv6, 3, 4,
                                  n_f*32, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv6, pool6 = transition(conv6, 3, n_f*32, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
   #################################################################################################
    
    # up stream
	up6 = Conv2DTranspose(n_f*16, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
	up6 = BatchNormalization(axis=3)(up6)
	up6 = Activation('relu')(up6)
	
	x1 = Reshape(target_shape=(np.int32(N/16), np.int32(N/16), n_f*16))(conv5)
	x2 = Reshape(target_shape=(np.int32(N/16), np.int32(N/16), n_f*16))(up6)
	merge6  = concatenate([x1,x2], axis = 3) 

	conv7, nb_filter = denseblock(merge6, 3, 4,
                                  n_f*8, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv7, pool7 = transition(conv7, 3, n_f*8, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
	conv7, nb_filter = denseblock(conv7, 3, 4,
                                  256, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv7, pool7 = transition(conv7, 3, n_f*8, dropout_rate=None,
                       weight_decay=1E-4, pool=False)	
    ########################################################
	up5 = Conv2DTranspose(n_f*8, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
	up5 = BatchNormalization(axis=3)(up5)
	up5 = Activation('relu')(up5)
	print(conv4.shape)
	print(up5.shape)	
	x1 = Reshape(target_shape=(np.int32(N/8), np.int32(N/8), n_f*8))(conv4)
	x2 = Reshape(target_shape=(np.int32(N/8), np.int32(N/8), n_f*8))(up5)
	merge7  = concatenate([conv4,up5], axis = 3) 
     
	conv8, nb_filter = denseblock(merge7, 3, 4,
                                  n_f*4, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv8, pool8 = transition(conv8, 3, n_f*4, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
	conv8, nb_filter = denseblock(conv8, 3, 4,
                                  n_f*4, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv8, pool8 = transition(conv8, 3, n_f*4, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up4 = Conv2DTranspose(n_f*4, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv8)
	up4 = BatchNormalization(axis=3)(up4)
	up4 = Activation('relu')(up4)
	
	x1 = Reshape(target_shape=(np.int32(N/4), np.int32(N/4), n_f*4))(conv3)
	x2 = Reshape(target_shape=(np.int32(N/4), np.int32(N/4), n_f*4))(up4)
	merge8  = concatenate([conv3,up4], axis = 3) 
       
	conv9, nb_filter = denseblock(merge8, 3, 4,
                                  n_f*2, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv9, pool9 = transition(conv9, 3, n_f*2, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
	conv9, nb_filter = denseblock(conv9, 3, 4,
                                  n_f*2, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv9, pool9 = transition(conv9, 3, n_f*2, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up3 = Conv2DTranspose(n_f*2, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv9)
	up3 = BatchNormalization(axis=3)(up3)
	up3 = Activation('relu')(up3)
	
	x1 = Reshape(target_shape=(np.int32(N/2), np.int32(N/2), n_f*2))(conv2)
	x2 = Reshape(target_shape=(np.int32(N/2), np.int32(N/2), n_f*2))(up3)
	merge9  = concatenate([conv2,up3], axis = 3) 
	#merge9 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge9)
            
	conv10, nb_filter = denseblock(merge9, 3, 4,
                                  n_f*1, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv10, pool10 = transition(conv10, 3, n_f*1, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
	conv10, nb_filter = denseblock(conv10, 3, 4,
                                  n_f*1, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv10, pool10 = transition(conv10, 3, n_f*1, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	up2 = Conv2DTranspose(n_f, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv10)
	up2 = BatchNormalization(axis=3)(up2)
	up2 = Activation('relu')(up2)
	
	x1 = Reshape(target_shape=(np.int32(N), np.int32(N), n_f))(conv1)
	x2 = Reshape(target_shape=(np.int32(N), np.int32(N), n_f))(up2)
	merge10  = concatenate([conv1,up2], axis = 3) 
        
	conv11, nb_filter = denseblock(merge10, 3, 4,
                                  n_f, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv11, pool11 = transition(conv11, 3, n_f, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
	conv11, nb_filter = denseblock(conv11, 3, 4,
                                  n_f, 12, 
                                  dropout_rate=None,
                                  weight_decay=1E-4)
	conv11, pool11 = transition(conv11, 3, n_f, dropout_rate=None,
                       weight_decay=1E-4, pool=False)
    ########################################################
	out_1 =  Conv2D(2, (1, 1), activation='softmax', name='final_1')(conv11)
	out_2 =  Conv2D(2, (1, 1), activation='softmax', name='final_2')(conv11)
	out_3 =  Conv2D(2, (1, 1), activation='softmax', name='final_3')(conv11)

	model = Model(inputs=[inputs], outputs=[out_1, out_2, out_3])

	loss = {'final_1': binary_crossentropy,
            'final_2': binary_crossentropy,
            'final_3': binary_crossentropy}

	loss_weights = {'final_1': 1,
                    'final_2': 1,
                    'final_3': 1}

	model.compile(optimizer=opt, loss=loss, metrics=[dsc, tp, tn], loss_weights=loss_weights)  
	return model

# opt = SGD(lr=0.01, momentum=0.90, decay=1e-6)
# model = DenseUnet2(opt, input_size=(256, 256, 3))
# model.summary()

def unet_3_out_right(opt, input_size = (256,256,1)):
    n_f = 16
    inputs = Input(input_size)
    conv1 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(drop5)
    
    conv55 = Conv2D(n_f*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv55 = Conv2D(n_f*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv55)
    drop55 = Dropout(0.5)(conv55)

    up6 = Conv2D(n_f*16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop55))
    merge6 = concatenate([drop5,up6], axis = 3)
    conv6 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(n_f*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(n_f*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([drop4,up7], axis = 3)
    conv7 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(n_f*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(n_f*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([drop3,up8], axis = 3)
    conv8 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(n_f*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2D(n_f*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv2,up9], axis = 3)
    conv9 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(n_f*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    up10 = Conv2D(n_f, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    merge10 = concatenate([conv1,up10], axis = 3)
    conv10 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = Conv2D(n_f, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)

    ######
    out_1 =  Conv2D(2, (1, 1), activation='softmax', name='final_1')(conv10)
    out_2 =  Conv2D(2, (1, 1), activation='softmax', name='final_2')(conv10)
    out_3 =  Conv2D(2, (1, 1), activation='softmax', name='final_3')(conv10)

    model = Model(inputs=[inputs], outputs=[out_1, out_2, out_3])

    loss = {'final_1': binary_crossentropy,
            'final_2': binary_crossentropy,
            'final_3': binary_crossentropy}

    loss_weights = {'final_1': 1,
                    'final_2': 1,
                    'final_3': 1}

    model.compile(optimizer=opt, loss=loss, metrics=[dsc, tp, tn], loss_weights=loss_weights)
    ######
    return model
