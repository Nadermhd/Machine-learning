# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:19:00 2019

@author: aldojn
"""

import numpy as np
import json

def config_():
    config = {
            'IMG_SIZE1': 256, 
            'IMG_SIZE2': 256,
            'SLICE_COUNT': 1,
            'num_epochs': 100,
            'n_epoch': 100,
            'batch_norm': True,
            'sce_weight': 1.0,
            'learnin_rate_beg': 1e-3,
            'learnin_rate_end': 1e-4,
            'n_epoch': 100,
            'data_dir_training': 'D:\\Work\\mre_t2\\clean_all_seq_new\\training',
            'data_dir_test': 'D:\\Work\\mre_t2\\clean_all_seq_new\\test',
            'eval_op': None,
            'gpu_ind': '1',
            'flag_resume': False,
            'which_seq': 'adc_',
            'which_run': 1,
            'beg': 0,
            'in_ch': 3,
            'lbl_ch': 4,
            'max_to_keep': 60,
            'b_size': 1,
            'batch_size_v': 25,
            'n_samples' : 775,
            'val_n': 50
            }
    return config

#json.dumps(config_(), indent = 4)

with open('conf.json', 'w') as outfile:
    json.dump(config_(), outfile)