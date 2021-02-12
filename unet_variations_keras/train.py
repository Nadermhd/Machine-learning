import os
os.environ['CUDA_VISISBLE_DEVICES'] = '0'
import conv_lstm as M
import numpy as np 
from keras.callbacks import ModelCheckpoint, Tensorboard, ReduceLROnPlateau
from keras import callbacks
import pickle

# ----------Normalize over the dataset
def dataset_normalized(imgs):

	imgs_normalized = np.empty(imgs.shape)
	imgs_std = np.std(imgs)
	imgs_mean = np.mean(imgs)
	imgs_normalized = (imgs-imgs_mean)/imgs_std
	# normalize images one by one
	for i in range(imgs.shape[0]):
		imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / np.max(imgs_normalized[i] - np.min(imgs_normalized[i]))) * 255

	return imgs_normalized

def mini_batches(n,X,Y):

    Nsamples=np.shape(X)[0]
    I = np.arange(0,Nsamples)
    n_per_mini_batch = Nsamples//n
    return [[X[I[k*n_per_mini_batch:(k+1)*n_per_mini_batch]],Y[I[k*n_per_mini_batch:(k+1)*n_per_mini_batch]]] for k in range(0,n)]

def DataSet_reshape(N, train_list):
    XX=np.zeros((len(train_list), IMG_SIZE1*IMG_SIZE2*2))
    YY=np.zeros((len(train_list), IMG_SIZE1*IMG_SIZE2*6))
    for i in range(len(train_list)):
        XX[i,:]=np.reshape(train_list[i][0], [IMG_SIZE1*IMG_SIZE2*2])
        YY[i,:]=np.reshape(train_list[i][1], [IMG_SIZE1*IMG_SIZE2*6])
    return XX,YY

####################################  Load Data #####################################

data_dir = '/scratch/Downloads/unet/unet_prostate_All_images_t2/new_runs_prostate_15tests'

tr_data = np.load(data_dir + '/in_images.npy')
tr_mask = np.load(data_dir + '/in_lables.npy')
print('------------------------dataset is loaded----------------------------')
print(np.shape(tr_data), np.shape(tr_mask))

num_data = np.shape(ts_data[0])
num_val = 400

tr_imgs = tr_data[:num_data - num_val]
tr_msks = tr_mask[:num_data - num_val]
val_data = tr_data[num_data - num_val:]
val_mask = tr_mask[num_data - num_val:]

tr_im = np.reshape(tr_imgs, [num_data - num_val, 256, 256, 2])
tr_ms = np.reshape(tr_msks, [num_data - num_val, 256, 256, 6])
va_im = np.reshape(val_data, [num_val, 256, 256, 2])
va_ms = np.reshape(val_mask, [num_val, 256, 256, 6])

tr_i = tr_im[: ,:, :, :1]
tr_m = tr_ms[:, :, :, :1]
va_i = va_im[:, :, :, :1]
va_m = va_ms[:, :, :, :1]
print('split the dataset')
print(np.shape(tr_im),np.shape(tr_i))
# split the train into train and vald

ts_data = np.load(data_dir + '/in_images_tst.npy')
tr_mask = np.load(data_dir + '/in_lables_tst.npy')


#################################### Buid model
model = M.BDLSTM_unet_Dense(input_size=(256, 256, 3))
model.summary()

batch_size = 8
nb_epoch = 100

mcp_save = ModelCheckpoint('weight_prostate', save_best_only=True, monitor='val_loss',mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
history = model.fit(tr_data,
	tr_mask,
	batch_size=batch_size,
	epochs=nb_epoch,
	shuffle=True,
	verbose=1,
	validation_data=(val_data, val_mask), callbacks=[mcp_save, reduce_lr_loss])

print('Trained Model saved')
# This is to save the log
with open('hist_prostate', 'wb') as file_pi:
	pickle.dump(history.history, file_pi)