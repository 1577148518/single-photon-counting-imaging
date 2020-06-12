# -*- coding=utf-8 -*-
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


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
		
def forward(x,regularizer):
	#x = tf.reshape(x,[-1,FLAGS.image_size,FLAGS.image_size,1])
	#print(x.shape)
	
	conv1_w = get_weight([FLAGS.conv_core,FLAGS.conv_core,1,FLAGS.sample_rate],FLAGS.REGULARIZER)
	conv1_w_B = quantize(conv1_w)
	conv1 = conv2d(x,conv1_w,FLAGS.conv_core)
	print(conv1)
	
	conv1s_shape = tf.shape(conv1)
	
	deconv1_w = get_weight([4,4,FLAGS.sample_rate,FLAGS.sample_rate],FLAGS.REGULARIZER)
	deconv1 = tf.nn.conv2d_transpose(value=conv1, filter=deconv1_w, output_shape=[conv1s_shape[0], 8, 8, FLAGS.sample_rate], strides=[1, 2, 2, 1], padding='SAME', name='deconv_1')
	deconv1 = tf.nn.relu(deconv1)
	
	deconv2_w = get_weight([4,4,int(FLAGS.sample_rate/2),FLAGS.sample_rate],FLAGS.REGULARIZER)
	deconv2 = tf.nn.conv2d_transpose(value=deconv1, filter=deconv2_w, output_shape=[conv1s_shape[0], 16, 16, int(FLAGS.sample_rate/2)], strides=[1, 2, 2, 1], padding='SAME', name='deconv_2')
	deconv2 = tf.nn.relu(deconv2)
	
	deconv3_w = get_weight([4,4,int(FLAGS.sample_rate/4),int(FLAGS.sample_rate/2)],FLAGS.REGULARIZER)
	deconv3 = tf.nn.conv2d_transpose(value=deconv2, filter=deconv3_w, output_shape=[conv1s_shape[0], 32, 32, int(FLAGS.sample_rate/4)], strides=[1, 2, 2, 1], padding='SAME', name='deconv_3')
	deconv3 = tf.nn.relu(deconv3)
	
	deconv4_w = get_weight([4,4,int(FLAGS.sample_rate/16),int(FLAGS.sample_rate/4)],FLAGS.REGULARIZER)
	deconv4 = tf.nn.conv2d_transpose(value=deconv3, filter=deconv4_w, output_shape=[conv1s_shape[0], 64, 64, int(FLAGS.sample_rate/16)], strides=[1, 2, 2, 1], padding='SAME', name='deconv_4')
	deconv4 = tf.nn.relu(deconv4)
	
	deconv5_w = get_weight([4,4,1,int(FLAGS.sample_rate/16)],FLAGS.REGULARIZER)
	deconv5 = tf.nn.conv2d_transpose(value=deconv4, filter=deconv5_w, output_shape=[conv1s_shape[0], 128, 128, 1], strides=[1, 2, 2, 1], padding='SAME', name='deconv_5')
	deconv5 = tf.nn.relu(deconv5)
	print(deconv5)
	#conv3_w = get_weight([32,32,int(FLAGS.sample_rate/128),1],FLAGS.REGULARIZER)
	#conv3_b = get_bias([1])
	#conv3 = tf.nn.conv2d(deconv5,conv3_w,strides=[1,1,1,1],padding='SAME')
	#y = tf.nn.bias_add(conv3,conv3_b)
	#print(y)
	
	return deconv5
	#num = int((FLAGS.image_size-FLAGS.conv_core)/FLAGS.conv_core + 1)
	
	#conv1 = tf.reshape(conv1,[-1,num*num,FLAGS.sample_rate])
	#print(conv1)
	#print(sess.run(conv1.eval()))
	#matmul2_w = get_weight([FLAGS.sample_rate,FLAGS.conv_core*FLAGS.conv_core],FLAGS.REGULARIZER)	
	#matmul2_b = get_bias([FLAGS.conv_core*FLAGS.conv_core])
	
	#out = np.zeros((num, num))
	#out = []
	#for x in range(num): 
	#	for y in range(num): 
	#		print(x,y)
	#		print(tf.slice(conv1,[-1,x,y,0],[-1,1,1,512]))
	#		m = tf.slice(conv1,[-1,x,y,0],[-1,1,1,512])
	#		m = tf.reshape(m,[-1,FLAGS.sample_rate])
	#		m = tf.matmul(m,matmul2_w) + matmul2_b
	#		m = tf.reshape(m,[-1,FLAGS.conv_core,FLAGS.conv_core])
	#		print(m)
	#		out.append(m)
	#		#out[x][y] = tf.slice(conv1,[-1,x,y,0],[-1,1,1,512])
	#		#out[x][y] = tf.reshape(m,[-1,FLAGS.sample_rate])
	#		#out[x][y] =tf.matmul(out[x][y],matmul2_w) + matmul2_b
	#		#out[x][y] = tf.reshape(m,[-1,FLAGS.conv_core,FLAGS.conv_core])
	#		print(out[x*num+y])
	#
	#n = tf.stack([out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],out[8],out[9],out[10],out[11],out[12],out[13],out[14],out[15]])
	#
	#n = tf.reshape(n,[-1,FLAGS.image_size,FLAGS.image_size])
	#print(n)
	#y = tf.matmul(conv1,matmul2_w) + matmul2_b
	#
	#for x in range(num): 
	#	print(x)
	#	out[x*num:(x+1)*num] = tf.matmul(conv1[-1,x],matmul2_w) + matmul2_b
	#matmul2_b = get_bias([FLAGS.image_size*FLAGS.image_size])
	#
	#y = tf.matmul(conv1[-1],matmul2_w)#+matmul2_b    #做乘法时自动少一维
	#y = tf.reshape(y,[-1,FLAGS.label_size,FLAGS.label_size])
	#print(y)
	
	
