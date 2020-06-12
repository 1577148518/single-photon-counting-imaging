# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

filters = [1, 64, 32, 1]
core_sizes = [11,1,7]

def get_weight(shape,regularizer):
	w = tf.Variable(tf.random_normal(shape, stddev=1e-3))
	if regularizer!= None: tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def matmul(x,w,b):
	return tf.nn.relu(tf.matmul(x,w) + b)

def rec_conv(x,core_size,in_kernel,out_kernel):
	conv_w = get_weight([core_size,core_size,in_kernel,out_kernel],FLAGS.REGULARIZER)
	conv_b = get_bias([out_kernel])
	conv = tf.nn.conv2d(x,conv_w,strides=[1,1,1,1], padding='SAME')
	relu = tf.nn.relu(tf.nn.bias_add(conv,conv_b))
	return relu
	
def Recon(x):
	print("x:",x)
	x1 = rec_conv(x,core_sizes[0],filters[0],filters[1])
	print("x1:",x1)
	x2 = rec_conv(x1,core_sizes[1],filters[1],filters[2])
	print("x2:",x2)
	x3 = rec_conv(x2,core_sizes[2],filters[2],filters[3])
	print("x3:",x3)
	
	return x3


def forward(x,regularizer):
	x = tf.reshape(x,[-1,FLAGS.image_size*FLAGS.image_size])
	
	matmul1_w = get_weight([FLAGS.image_size*FLAGS.image_size,FLAGS.compress_size],FLAGS.REGULARIZER)	
	matmul1_b = get_bias([FLAGS.compress_size])
	matmul1 = matmul(x,matmul1_w,matmul1_b)
	
	matmul2_w = get_weight([FLAGS.compress_size,FLAGS.image_size*FLAGS.image_size],FLAGS.REGULARIZER)	
	matmul2_b = get_bias([FLAGS.image_size*FLAGS.image_size])
	y = tf.matmul(matmul1,matmul2_w)+matmul2_b
	
	y = tf.reshape(y,[-1,FLAGS.image_size,FLAGS.image_size,1])
	
	rec = Recon(y)
	
	return rec 
	
	
