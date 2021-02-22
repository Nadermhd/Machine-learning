#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:08:02 2021

@author: nader
"""

import Models_3_out as M
from keras.optimizers import SGD 
from data_class import dataset_
from train_class import train_valid_test
import argparse
import os
from datetime import datetime
import numpy as np
from save_load import save_models, load_models


PARSER = argparse.ArgumentParser(description='Unet-medical-dominan-adaptation')

PARSER.add_argument('--main_data_dir',
	type=str,
	default='/media/nader/WinD/Work/data/newsets/Brain',
	help='data directory ')

PARSER.add_argument('--sub_dir_train',
	type=str,
	default='imagesTr',
	help='sub data directory for training')

PARSER.add_argument('--labels',
	type=str,
	default='labelsTr',
	help='sub data directory for training')

PARSER.add_argument('--sub_dir_test',
	type=str,
	default='imagesTs',
	help='sub data directory for testing')

PARSER.add_argument('--n_epochs',
	type=int,
	default='10',
	help='No of total epochs')

PARSER.add_argument('--batch_size',
	type=int,
	default='5',
	help='batch size')

PARSER.add_argument('--train_num',
	type=int,
	default='484',
	help='Total num of patients')

PARSER.add_argument('--precentage',
	type=float,
	default='0.6',
	help='precentage of train - valid ')

PARSER.add_argument('--lr',
	type=float,
	default='1e-4',
	help='learning rate')

PARSER.add_argument('--seed',
	type=float,
	default='123',
	help='rando seed')

PARSER.add_argument('--size',
	type=tuple,
	default=(240, 240, 155),
	help='original image size')

PARSER.add_argument('--s_s_ch', # to be changed for domian adaptation 
	type=tuple,
	default=(240, 240, 4), 
	help='size, size, ch')

PARSER.add_argument('--train', # to be changed for domian adaptation 
	type=bool,
	default=True, 
	help='training flag')

PARSER.add_argument('--out_dir', # to be changed for domian adaptation 
	type=str,
	default='/media/nader/WinD/Work/data/newsets/Brain/out', 
	help='output dir')

PARSER.add_argument('--expirment_dir', # to be changed for domian adaptation 
	type=str,
	default='', 
	help='Experiment dir')

args = PARSER.parse_args()

def main(): 
    
    # load data patient per pateint
    d = dataset_(1, args.main_data_dir, 1, args.main_data_dir, seed=args.seed, size=args.size) 
    ts = d.create_testing_abtImage()
    im , lb, slices = d.create_training_validating_data_abtImage(0)
    
    # Get the model
    opt = SGD(lr=args.lr, momentum=0.90, decay=1e-6)
    model = M.unet_normal(opt,input_size=args.s_s_ch)
    
    ID = str(datetime.now().date()) + '-' + str(datetime.now().hour) + '-' + str(datetime.now().minute) + str(datetime.now().second)
            
    if args.train:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        args.expirment_dir = os.path.join(args.out_dir, str(ID))
        os.mkdir(args.expirment_dir)

        # Train and validate
        h_tr, h_vl = train_valid_test(d, n_epochs=args.n_epochs, batch_size=args.batch_size,
                        im=im, lb=lb, model=model, precentage = args.precentage, out_dir=args.expirment_dir).train()
        
        np.save(os.path.join(args.expirment_dir, 'h_tr'), h_tr)
        np.save(os.path.join(args.expirment_dir, 'h_vl'), h_vl)
        
        # test after training
        h_ts = train_valid_test(d, n_epochs=args.n_epochs, batch_size=args.batch_size, 
                        im=ts, lb=[], model=model, precentage = args.precentage, out_dir=args.expirment_dir).test() 
        np.save(os.path.join(args.expirment_dir, 'h_ts'), h_ts)
    
    # only testing without training
    else: 
        args.expirment_dir = '' # copy and paste specific directory
        
        weights = args.expirment_dir + '/model_%03d_w.h5' % (10) # change epoch num
        model.load_weights(weights)
        
        h_ts = train_valid_test(d, n_epochs=args.n_epochs, batch_size=args.batch_size, 
                        im=ts, lb=[], model=model, precentage = args.precentage, out_dir=args.expirment_dir).test() 
        
  
if __name__ == '__main__':
    main()
    
# what is left: - save the model, and build the test function
