#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:17:06 2021

@author: nader
"""

import keras

def save_models(step, model):
    # save the first generator model
    filename1 = 'model_%06d_m.h5' % (step)
    filename2 = 'model_%06d_w.h5' % (step)
    model.save(filename1)
    model.save_weights(filename2)

    print('>Saved: %s ' % (filename1))
    
def load_models(path, step, model):
    # save the first generator model
    weights = path + 'model_%06d_w.h5' % (step)
    m = path + 'model_%06d_m.h5' % (step)

    model.load(weights)
    model.load_weights(weights)
    
    print('>Loaded: %s ' % (weights))
    return 