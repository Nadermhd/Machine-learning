import numpy as np 
import pandas as pd
import keras
import SimpleITK as sitk 
import cv2

import imgaug as ia 
from imgaug import augmenters as iaa


class DataGen_(keras.utils.Sequence):
	def __init__(self, imgs, lbls1, lbls2, lbls3, batch_size=32, image_dimensions=(256, 256, 3), shuffle=False, augment=False):
		self.imgs = imgs
		self.lbls1 = lbls1
		self.lbls2 = lbls2
		self.lbls3 = lbls3
		self.dim = image_dimensions
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.augment = augment
		self.on_epoch_end()
        
		print(len(self.imgs))

	def __len__(self):
		return int(np.floor(len(self.imgs)/self.batch_size))

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.imgs))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]
		print(indexes)
		lables1 = np.array([self.lbls1[k] for k in indexes])
		lables2 = np.array([self.lbls2[k] for k in indexes])
		lables3 = np.array([self.lbls3[k] for k in indexes])
		images = np.array([self.imgs[k] for k in indexes])
		#print(np.shape(images))

		if self.augment:
			imall = np.zeros((self.batch_size, self.dim[0], self.dim[1], 9))
			imall[..., 0:3] = images
			imall[..., 3:5] = lables1
			imall[..., 5:7] = lables2
			imall[..., 7:9] = lables3
            
			imagesall = self.augmentor(imall)

		return imagesall[..., 0:3], [imagesall[..., 3:5], imagesall[..., 5:7], imagesall[..., 7:9]]

	def augmentor(self, images):
		sometimes = lambda aug:iaa.Sometimes(0.5, aug)
		seq = iaa.Sequential([
			sometimes(iaa.ElasticTransformation(alpha=(0.0, 30.0), sigma=5.0)),
			sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1))),
			sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
			sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
			], random_order=True)
# 		seq = iaa.Sequential([
#  			iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
#  			sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
# 			sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
# 			], random_order=True)
        
		return seq.augment_images(images)

