import numpy as np
import tensorflow as tf

import os
import time
import cv2
import random

from tools import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  make_curve
)
from forward import forward
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
def backward(nx,ny):
	
	images = tf.placeholder(tf.float32, [
			None,
			FLAGS.image_size,FLAGS.image_size,1])
	labels = tf.placeholder(tf.float32, [
			None,
			FLAGS.label_size,FLAGS.label_size,1])
			
	data_dir = os.path.join('./{}'.format(FLAGS.checkpoint_dir), "train.h5")
	data_dir_test = os.path.join('./{}'.format(FLAGS.checkpoint_dir), "test.h5")
	
	train_data, train_label = read_data(data_dir)
	test_data,test_label = read_data(data_dir_test)
			
	pred = forward(images,FLAGS.REGULARIZER)
	
	global_step = tf.Variable(0,trainable=False)
	
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.square(labels - pred))
		#loss = loss + tf.add_n(tf.get_collection('losses'))	
		loss_summary = tf.summary.scalar('loss', loss)
	
	loss_sequence = []
	valloss_sequence = []
	counter_sequence = []
	
	merged = tf.summary.merge_all()
	
	saver = tf.train.Saver()
	
	index_ = [i for i in range(len(train_data))]
	#***********************************************************
	#print(len(train_data))
	#*********************************************************** 
	
	Learning_rate = tf.train.exponential_decay(
		FLAGS.learning_rate,
		global_step,
		len(train_data)/FLAGS.batch_size,
		FLAGS.learding_rate_delay,
		staircase = True)
		
	train_step = tf.train.AdamOptimizer(Learning_rate).minimize(loss,global_step = global_step)
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
	
		counter = 0
		start_time = time.time()
		
		checkpoint_dir = load(FLAGS.checkpoint_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			   	
		writer = tf.summary.FileWriter('save-Autoencoder/logs', sess.graph)
		
		if FLAGS.is_train:
			print("Training...")
			
			for ep in range(FLAGS.epoch):
    	    	# Run by batch images
				batch_idxs = len(train_data) // FLAGS.batch_size
				
				random.shuffle(index_)
				train_data = train_data[index_]
				train_label = train_label[index_]
						
				for idx in range(0, batch_idxs):
							
					batch_images = train_data[idx*FLAGS.batch_size : (idx+1)*FLAGS.batch_size]
					batch_labels = train_label[idx*FLAGS.batch_size : (idx+1)*FLAGS.batch_size]
						
					batch_images_test = test_data[0 : FLAGS.batch_size]
					batch_labels_test = test_label[0 : FLAGS.batch_size]
						
					counter += 1
					
					_,summary, err = sess.run([train_step,merged,loss], feed_dict={images: batch_images, labels: batch_labels})
					val_err = sess.run(loss,feed_dict={images: batch_images_test, labels: batch_labels_test})
					
					loss_sequence.append(err)
					valloss_sequence.append(val_err)
					counter_sequence.append(counter)
					
					writer.add_summary(summary, counter)
					
					if counter % 10 == 0:
						print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
						% ((ep+1), counter, time.time()-start_time, err))	
				
						make_curve(loss_sequence,valloss_sequence,counter_sequence)
						loss_sequence = []
						valloss_sequence = []
						counter_sequence = []
						
					if counter % 300 == 0:
						
						saver.save(sess,
                    		os.path.join(save(FLAGS.checkpoint_dir, counter),"Autoencoder.model"),
                    		global_step= counter)
						    		
		else:
			print("Testing...")
			result = pred.eval({images: test_data,labels: test_label})
			
			#result = merge(result, [nx, ny])
			result = merge(result, [nx, ny])
			result = result.squeeze()
			image_path = os.path.join(os.getcwd(), FLAGS.sample_dir)
			image_path = os.path.join(image_path, "test_image.png")
			result = result * 255
			result = result.astype(np.int32)
			imsave(image_path,result)
			#############################################################
			Label = test_label
			Label = merge(Label, [nx, ny])
			Label = Label.squeeze()
			image_path1 = os.path.join(os.getcwd(), FLAGS.sample_dir)
			image_path1 = os.path.join(image_path1, "orignal.png")
			imsave(image_path,result)
			#print(Label)
			Label = Label * 255
			Label = Label.astype(np.int32)
			imsave(image_path1,Label)
			
def save(checkpoint_dir, step):
    model_dir = "%s_%s" % ("Autoencoder", FLAGS.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
    if not os.path.exists(checkpoint_dir):
    	os.makedirs(checkpoint_dir)
    return checkpoint_dir

                    	
def load(checkpoint_dir):
	print(" [*] Reading checkpoints...")
	model_dir = "%s_%s" % ("Autoencoder",FLAGS.label_size)
	checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	return checkpoint_dir