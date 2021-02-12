# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:19:00 2019

@author: aldojn
"""
import tensorflow as tf

import os as os
import pandas as pd 
from scipy import ndimage
import matplotlib.pyplot as plt
from termcolor import colored, cprint
from tqdm import tqdm
from Helper_functions_unet_ab_mask import dataset_
from losses import pixel_wise_loss, dice_coef_loss, mask_prediction, soft_dice, focal_loss
from unet import unet
from unet_layers_dense import dense_conv_module, upsample
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
import numpy as np
from augment import rotate_img, flip_img, zoom_img, shift_img, elastic_transform

IMG_SIZE1 = 128 
IMG_SIZE2 = 128
batch_size = 16
learnin_rate = 1e-5
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
eval_op = None
num_epochs = 150
n_epoch = 150
batch_norm = True
sce_weight = 1. 
gpu_ind = '1'

flag_resume = False
beg = 0

exp_name = 'all_seq_' + str(learnin_rate) 

curr_dir = os.getcwd()
prev_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
path = os.path.join(prev_dir, 'results_normal')
path = os.path.join(path, exp_name)
os.chdir(os.path.join(prev_dir, 'results_normal'))

if not os.path.isdir(path):
    os.mkdir(path)
    os.chdir(path)
    os.chdir(os.path.join(path, 'out_1'))
    os.mkdir('train')
    os.mkdir('test')

save_dir_train_ = '/results_normal/' + exp_name + '/out_1/train'
save_dir_test_ = '/results_normal/' + exp_name + '/out_1/test'
save_dir_ = '/model_1'
tb_tr = '/results_normal/'+ exp_name + '/tmp_1/train'
tb_vl = '/results_normal/'+ exp_name + '/tmp_1/test'

data_dir_training='/training'
data_dir_test='/test_new_2'

mask_img_dice_all = []  
mask_img_di_all = []
mask_img_xe_all = []
mask_img_x_all = []
batch_X_t_all = []
batch_Y_t_all = []
batch_Y_t_all_pz = []
predc = []
mask_img_c = []

mean_dice = []
median_dice = []
std_dice = []
Mean_Rel_Abs_Vol = []
Mean_Hauss_Dist = []
Mean_MSD = []

def mini_batches(n,X,Y):
   
    Nsamples=np.shape(X)[0]
    I = np.arange(0,Nsamples)
   
    n_per_mini_batch = Nsamples//n

    return [[X[I[k*n_per_mini_batch:(k+1)*n_per_mini_batch]],Y[I[k*n_per_mini_batch:(k+1)*n_per_mini_batch]]] for k in range(0,n)]

def DataSet_reshape(N, train_list):
    XX=np.zeros((len(train_list), IMG_SIZE1*IMG_SIZE2*3))
    YY=np.zeros((len(train_list), IMG_SIZE1*IMG_SIZE2*4))
    for i in range(len(train_list)):
        XX[i,:]=np.reshape(train_list[i][0], [IMG_SIZE1*IMG_SIZE2*3])
        YY[i,:]=np.reshape(train_list[i][1], [IMG_SIZE1*IMG_SIZE2*4])
    return XX,YY

n_samples = 775 

N=725/1

b_size = 1

batch_size = n_samples/b_size
batch_size_v = 25



d = dataset_(50, data_dir_training, 50, data_dir_test) 

patients_training=os.listdir(data_dir_training)
patients_training.sort()

training_data=[]
print('loading Training Data...')
training_data = d.create_training_validating_data_abtImage()                                                                                                               
train = training_data[:n_samples-50] # 10100
val = training_data[-50:]
print('len train', len(training_data)) 
del training_data

xx,yy=DataSet_reshape(batch_size, train)
xx_v,yy_v=DataSet_reshape(batch_size_v, val)
print(np.shape(yy))
B = mini_batches(N,xx,yy)
B_v = mini_batches(2,xx_v,yy_v)
print('file saved')
del xx, yy


###########################THE U-NET MODEL ##############################
# place holders
roi_images = tf.placeholder(tf.float32, [None, IMG_SIZE1, IMG_SIZE2, 3], name = "roi_images")
roi_masks = tf.placeholder(tf.float32, [None, IMG_SIZE1, IMG_SIZE2, 4], name = "roi_masks")
roi_weights = tf.placeholder(tf.float32, [None, IMG_SIZE1, IMG_SIZE2], name = "roi_weights")
global_step = tf.Variable(0, name='global_step', trainable=False)

def build_net(training_flag):
        batch_norm = True
        
        mask_logits0, mask_logits1 = unet(roi_images, num_classes=4, training=training_flag,
                           init_channel=16, n_layers=6, batch_norm=batch_norm)
        return mask_logits0, mask_logits1
    
mask_logits0, mask_logits1 = build_net(training_flag)

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in 'conv_module_up1']

roi_masks_pos0 = roi_masks[:,:,:,0:2]
roi_masks_pos1 = roi_masks[:,:,:,2:4]


print('mask_logits', mask_logits0)


with tf.name_scope("loss_dice"):
    dice1 = dice_coef_loss(roi_masks_pos0[:,:,:,1], mask_logits0[:,:,:,1])
    dice2 = dice_coef_loss(roi_masks_pos1[:,:,:,1], mask_logits1[:,:,:,1])
    loss0 =  dice_coef_loss(roi_masks_pos0[:,:,:,1], mask_logits0[:,:,:,1]) + dice_coef_loss(roi_masks_pos1[:,:,:,1], mask_logits1[:,:,:,1])
    
with tf.name_scope("loss_Xent"):
    tf_mask0 = tf.cast(mask_logits0, tf.float32)
    tf_mask1 = tf.cast(mask_logits1, tf.float32)
    tf0 = pixel_wise_loss(tf_mask0, roi_masks_pos0, pixel_weights=None)
    tf1 = pixel_wise_loss(tf_mask1, roi_masks_pos1, pixel_weights=None)
    
    loss1 = (pixel_wise_loss(tf_mask0, roi_masks_pos0, pixel_weights=None) +pixel_wise_loss(tf_mask1, roi_masks_pos1, pixel_weights=None))/3.0
    
with tf.name_scope("loss_focal"):
    focal= focal_loss(mask_logits0[:,:,:,0], roi_masks_pos0[:,:,:,0]) + focal_loss(mask_logits0[:,:,:,0], roi_masks_pos0[:,:,:,0])
    
with tf.name_scope("loss"):
    loss = loss0 + sce_weight * loss1
    
with tf.name_scope("train"):
    solver = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
         train_op = solver.minimize(loss1)
         loss_op = [loss0, loss1]
         
tf.summary.scalar('Dice_p', -dice1)
tf.summary.scalar('Dice_pz', -dice2)


tf.summary.scalar('Xent_p', tf0)
tf.summary.scalar('Xent_pz', tf1)

tf.summary.image('label_p', tf.slice(roi_images, [0, 0, 0, 1], [-1, -1, -1, 1]))
tf.summary.image('label_p', tf.slice(roi_masks, [0, 0, 0, 1], [-1, -1, -1, 1]))
tf.summary.image('label_pz', tf.slice(roi_masks, [0, 0, 0, 1], [-1, -1, -1, 1]))
tf.summary.image('prediction', tf.slice(mask_logits0, [0, 0, 0, 1], [-1, -1, -1, 1]))
merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(tb_tr)
test_writer = tf.summary.FileWriter(tb_vl)

loss_pixel = []
loss_dice = []
loss_softdice = []

loss_pixel_t = []
loss_dice_t = []

total_parameters = 0
for variable in tf.trainable_variables():

    shape = variable.get_shape()

    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print('total_parameters', total_parameters/1000000)

config_gpu = tf.ConfigProto()
config_gpu.gpu_options.visible_device_list= gpu_ind

batch_file_X = np.zeros((10, IMG_SIZE1, IMG_SIZE1, 3))
batch_file_Y = np.zeros((10, IMG_SIZE1, IMG_SIZE1, 4))

def train(data):
    
    with tf.Session(config=config_gpu) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=60)

        train_writer.add_graph(sess.graph)
        test_writer.add_graph(sess.graph)
        
        if flag_resume:
            saver.restore(sess, tf.train.latest_checkpoint('./'))

        
        dice_loss, sce_loss, n_steps, tensorboard_step, focal_loss = 0, 0, 1, 0, 0
        
        for i_epoch in range(beg, n_epoch):


            print('*************************************** E-poch: ', i_epoch, '*********************************************')
            np.random.shuffle(B)
            
            save_dir_train = save_dir_train_
            for b0, b in enumerate(B[0:]):
                
                epoch_learning_rate = learnin_rate
                batch_X, batch_Y = np.reshape(b[0], [IMG_SIZE1, IMG_SIZE2, 3]), np.reshape(b[1], [IMG_SIZE1, IMG_SIZE2, 4]) 
                
                print('Sub_epoch', b0, ' of epoch', i_epoch)
                
                im_merge = np.concatenate((batch_X, batch_Y), axis=2)
                #print(np.shape(im_merge))    
                
                batch_file_X[0, :, :, :] = batch_X
                batch_file_Y[0, :, :, :] = batch_Y
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[1, :, :, :] = np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[1, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[2, :, :, :] = np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[2, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[3, :, :, :] = np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[3, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[4, :, :, :] = np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[4, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[5, :, :, :] =np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[5, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[6, :, :, :] = np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[6, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[7, :, :, :] = np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[7, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[8, :, :, :] = np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[8, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
                batch_file_X[9, :, :, :] = np.reshape(im_merge_t[...,0:3], [1, IMG_SIZE1, IMG_SIZE2, 3])
                batch_file_Y[9, :, :, :] = np.reshape(im_merge_t[...,3:], [1, IMG_SIZE1, IMG_SIZE2, 4])
                
                
                
                ops = train_op 
                train_feed_dict = {
	                roi_images: batch_file_X,
	                roi_masks: batch_file_Y, 
	                learning_rate: epoch_learning_rate,
	                training_flag : True
	            }
                #
                s = sess.run(merged_summary,feed_dict=train_feed_dict)
                train_writer.add_summary(s, tensorboard_step)
                tensorboard_step = tensorboard_step +1

                pred_prob0 = sess.run(mask_logits0, feed_dict=train_feed_dict)  # tf_mask
                pred_prob1 = sess.run(mask_logits1, feed_dict=train_feed_dict)
                mask_tf0 = sess.run(tf_mask0, feed_dict=train_feed_dict) 
                mask_tf1 = sess.run(tf_mask1, feed_dict=train_feed_dict)
                
                l0 = sess.run(loss0, feed_dict=train_feed_dict)
                loss_dice.append(l0)
                l1 = sess.run(loss1, feed_dict=train_feed_dict)
                loss_pixel.append(l1)
                l2 = sess.run(focal, feed_dict=train_feed_dict)

                #print('****************************', n_steps, '*******************************')
                #_, global_step, loss0, loss1 = sess.run(ops, feed_dict=train_feed_dict)
                xx = sess.run(ops, feed_dict=train_feed_dict) 

                if n_steps+1 == 20: # n_display needs to be assigned
                    print("Dice coeff: {}, Cross entropy: {}".format((dice_loss/n_steps)/3.0, sce_loss/n_steps))
                    dice_loss, sce_loss, n_steps, tb_step = 0, 0, 0, 0
                    
                else:
                    dice_loss += l0
                    sce_loss += l1
                    n_steps += 1
                    focal_loss = l2
                    
                save_dir_test = save_dir_test_
                
            if i_epoch % 2 == 0:
                
                save_dir = save_dir_
                name_graph = 'model_'+ str(i_epoch)
                saver.save(sess, os.path.join(save_dir,name_graph))
            # ---------------------------------------------------------------------------------- TESTING ------------------------------------------------------------# 
       
            print("******************{} epochs finished.**************************************".format(i_epoch))

train(train)
























