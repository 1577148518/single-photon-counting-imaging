# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

filters = [1, 16, 4, 1]

w_lay1 = [1,3]
w_lay2 = [1,5]
w_lay3 = [1,9]
w_lay4 = [1,11]
w_lay5 = [1,15]

kernals_lay1 = [1,32,32,1]
kernals_lay2 = [1,32,32,1]
kernals_lay3 = [1,32,32,1]
kernals_lay4 = [1,32,32,1]
kernals_lay5 = [1,32,32,1]

def get_weight(shape,regularizer):
	w = tf.Variable(tf.random_normal(shape, stddev=1e-3))
	if regularizer!= None: tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def matmul(x,w,b):
	return tf.nn.relu(tf.matmul(x,w) + b)

def conv2d(x,w,stride):
	return tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding='SAME')


@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
	return tf.clip_by_value(tf.identity(grad), -1, 1)
        
def quantize(x):
	with tf.get_default_graph().gradient_override_map({"Sign": "QuantizeGrad"}):
		x = tf.sign(x)
		return x

def res_conv(x,in_kernel,out_kernel):
	conv_w = get_weight([5,5,in_kernel,out_kernel],FLAGS.REGULARIZER)
	conv_b = get_bias([out_kernel])
	conv = tf.nn.conv2d(x,conv_w,strides=[1,1,1,1], padding='SAME')
	relu = tf.nn.relu(tf.nn.bias_add(conv,conv_b))
	return relu
	
def residual(x):
	orig_x = x
	
	x = res_conv(x,filters[0],filters[1])
	x = res_conv(x,filters[1],filters[2])
	x = res_conv(x,filters[2],filters[3])
	
	x += orig_x
	return x

def shift_ps(I,h,w):
    bsize, a, b, c = I.get_shape().as_list() #i是一个四维元组
    print("bsize_before",bsize)
    bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
    print("bsize_after",bsize)
    X = tf.reshape(I, (bsize, a, b, h, w))#变成了五维？
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1

    X = tf.split(X,a,1)# a, [bsize, b, r, r]

    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)
    print(X.shape)# bsize, b, a*r, r
    X = tf.split(X,b,1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
    X= tf.reshape(X, (bsize, a*h, b*w, 1))
    return X
    
def inception_conv(x,in_kernal,out_kernal,w_size,stride):
	conv_w = get_weight([w_size,w_size,in_kernal,out_kernal],FLAGS.REGULARIZER)
	conv_b = get_bias([out_kernal])
	return tf.nn.relu(tf.nn.conv2d(x,conv_w,strides=[1,stride,stride,1],padding='SAME')+conv_b)
	
def inception(x):
	conv1_1 = inception_conv(x,kernals_lay1[0],kernals_lay1[1],w_lay1[1],1)
	conv1_2 = inception_conv(conv1_1,kernals_lay1[1],kernals_lay1[2],w_lay1[0],1)
	conv1_3 = inception_conv(conv1_2,kernals_lay1[2],kernals_lay1[3],w_lay1[1],1)
	conv1_out = conv1_3
	
	conv2_1 = inception_conv(x,kernals_lay2[0],kernals_lay2[1],w_lay2[1],1)
	conv2_2 = inception_conv(conv2_1,kernals_lay2[1],kernals_lay2[2],w_lay2[0],1)
	conv2_3 = inception_conv(conv2_2,kernals_lay2[2],kernals_lay2[3],w_lay2[1],1)
	conv2_out = conv2_3
	
	conv3_1 = inception_conv(x,kernals_lay3[0],kernals_lay3[1],w_lay3[1],1)
	conv3_2 = inception_conv(conv3_1,kernals_lay3[1],kernals_lay3[2],w_lay3[0],1)
	conv3_3 = inception_conv(conv3_2,kernals_lay3[2],kernals_lay3[3],w_lay3[1],1)
	conv3_out = conv3_3
	
	conv4_1 = inception_conv(x,kernals_lay4[0],kernals_lay4[1],w_lay4[1],1)
	conv4_2 = inception_conv(conv4_1,kernals_lay4[1],kernals_lay4[2],w_lay4[0],1)
	conv4_3 = inception_conv(conv4_2,kernals_lay4[2],kernals_lay4[3],w_lay4[1],1)
	conv4_out = conv4_3
	
	conv5_1 = inception_conv(x,kernals_lay5[0],kernals_lay5[1],w_lay5[1],1)
	conv5_2 = inception_conv(conv5_1,kernals_lay5[1],kernals_lay5[2],w_lay5[0],1)
	conv5_3 = inception_conv(conv5_2,kernals_lay5[2],kernals_lay5[3],w_lay5[1],1)
	conv5_out = conv5_3
	
	return conv1_out+conv2_out+conv3_out+conv4_out+conv5_out
	
def forward(x,regularizer):
	conv1_w = get_weight([FLAGS.conv_core,FLAGS.conv_core,1,FLAGS.sample_times],FLAGS.REGULARIZER)
	conv1 = conv2d(x,conv1_w,FLAGS.conv_core)
	print("conv1:",conv1)
	
	pixel1 = shift_ps(conv1,10,10)             # h*w等于采样次数
	print("pixel1",pixel1)
	
	recov_conv1_w = get_weight([10,10,1,1024],FLAGS.REGULARIZER)            #
	recov_conv1_b = get_bias([1024])
	recov_conv1 = tf.nn.relu(tf.nn.conv2d(pixel1,recov_conv1_w,strides=[1,10,10,1], padding='VALID')+recov_conv1_b)    #
	print("recov_conv1:",recov_conv1)
	
	pixel2 = shift_ps(recov_conv1,32,32)
	print("pixel2",pixel2)
	
	#y1 = inception(pixel2)
	#y1 = pixel2 + y1
	
	#y2 = inception(y1)
	#y2 = y1 + y2
	
	#y = pixel2 + y2
	return pixel2