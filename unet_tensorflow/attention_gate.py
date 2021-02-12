# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:19:00 2019

@author: aldojn
"""
import tensorflow as tf
import numpy as np


def attention_gate_fun(gating,features):


	shape = gating.get_shape().as_list()
	n_filters = shape[3]

	g = tf.layers.conv2d(gating, n_filters, (3, 3), activation=None, strides=(1,1), padding='same', name="gating_{}".format(i+1))
	x = tf.layers.conv2d(features, n_filters, (3, 3), activation=None, strides=(1,1), padding='same', name="features_{}".format(i+1))

	Sum = g + x

	Relu = tf.nn.relu(Sum)

	psi =  tf.layers.conv2d(features, n_filters, (3, 3), activation=None, strides=(1,1), padding='same', name="psi_{}".format(i+1))

	sigmoid = tf.sigmoid(psi) #tf.nn.sigmoid(psi)

	alpha = tf.Variable(tf.zeros([1]), trainable=True)

	resampler = tf.matmul(alpha,sigmoid)

	out = tf.matmul(resampler, x)

