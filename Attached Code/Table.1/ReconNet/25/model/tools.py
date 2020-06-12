# -*- coding=utf-8 -*-
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

import tensorflow as tf

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import cv2
import pandas as pd

FLAGS = tf.app.flags.FLAGS

def make_dir():
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)
    	
def make_data(data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')
    
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

def make_curve(loss,val_loss,counter):
	path = os.path.join(os.getcwd(), 'loss_valloss/curve.h5')
	
	loss = pd.Series(loss)
	val_loss = pd.Series(val_loss)
	count_ = pd.Series(counter)

	store = pd.HDFStore(path,'a')
	store.append('loss', loss,format='table')#,append=True)
	store.append('val_loss', val_loss,format='table')#,append=True)
	store.append('count', count_,format='table')#,append=True)
	store.close()
	  
def read_data(path):
    """
    Read h5 format data file
    
    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label
#读写h5文件

def prepare_data(dataset):
    """
    Args:
      dataset: choose train dataset or test dataset
      
      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    
    return data					#读取预设路径中指定文件夹的bmp格式图片，并以列表形式返回
    
def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)
      #将读取的图片转化成array格式，得到的是黑白图像

def imsave(path,image):
	cv2.imwrite(path,image)
    #return scipy.misc.toimage(image,)imsave(path, image) #保存图片
   
def preprocess(path):
    """
    Preprocess single image file 
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation
    
    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    input_ = imread(path, is_grayscale=True)         #读取Ycbcr图片，并返回数组
    
    input_ = input_ / 255.
    label_ = input_
    
    return input_, label_

#def merge(images, size):
#	#h = FLAGS.label_size
#	#w = h
#	print("***************************************")
#	#img = np.zeros((h*size[0], w*size[1], 1))
#	#for idx, image in enumerate(images):
#	image = images.reshape(FLAGS.label_size,FLAGS.label_size,1)
#	#i = idx % size[1]
#	#j = idx // size[1]
#	#img[j*h:j*h+h, i*w:i*w+w, :] = image
#	return image
	
def merge(images, size):
	h = FLAGS.label_size
	w = h
	print("***************************************")
	img = np.zeros((h*size[0], w*size[1], 1))
	for idx, image in enumerate(images):
		image = image.reshape(FLAGS.label_size,FLAGS.label_size,1)
		i = idx % size[1]
		j = idx // size[1]
		img[j*h:j*h+h, i*w:i*w+w, :] = image
	return img
	
def input_setup(config):
	
	if config.is_train:
		data = prepare_data(dataset="Train")
	else:
  		data = prepare_data(dataset="Test")
	
	sub_input_sequence = []
	sub_label_sequence = []
	
	if config.is_train:
		for i in range(len(data)):
			input_, label_ = preprocess(data[i])
			if len(input_.shape) == 3:
				h, w, _ = input_.shape
			else:
				h, w = input_.shape
			
			for x in range(0, h-config.image_size+1, config.stride):
				for y in range(0, w-config.image_size+1, config.stride):
					sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [28 x 28]
					sub_label = label_[x:x+config.label_size, y:y+config.label_size] # [28 x 28]
					
					sub_input = sub_input.reshape([config.image_size,config.image_size,1])  
					sub_label = sub_label.reshape([config.label_size,config.label_size,1])
					
					sub_input_sequence.append(sub_input)
					sub_label_sequence.append(sub_label)
			#sub_input_sequence = np.array(sub_input_sequence)
			#sub_label_sequence = np.array(sub_label_sequence)
			#print(sub_input_sequence.shape)
			#print(sub_label_sequence.shape)
	else:
		input_, label_ = preprocess(data[2])
		print(np.array(input_).shape)

		if len(input_.shape) == 3:
			h, w, _ = input_.shape
		else:
			h, w = input_.shape
		nx = ny = 0 
		for x in range(0, h-config.image_size+1,config.image_size):
			nx += 1; ny = 0
			for y in range(0, w-config.image_size+1,config.image_size):
				ny += 1
				sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [28 x 28]
				sub_label = label_[x:x+config.label_size, y:y+config.label_size] # [28 x 28]
        
				sub_input = sub_input.reshape([config.image_size,config.image_size,1])  
				sub_label = sub_label.reshape([config.label_size,config.label_size,1])

				sub_input_sequence.append(sub_input)
				sub_label_sequence.append(sub_label)
				#print(x,y)

	arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
	arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

	make_data(arrdata, arrlabel)

	if not config.is_train:
		return nx, ny
    
#def input_setup(sess, config):
#  """
#  Read image files and make their sub-images and saved them as a h5 file format.
#  """
#  # Load data path
#  if config.is_train:
#    data = prepare_data(sess, dataset="Train")
#  else:
#    data = prepare_data(sess, dataset="Test")
#
#  sub_input_sequence = []
#  sub_label_sequence = []
#  padding = abs(config.image_size - config.label_size) / 2 # 6             //等比缩小
#
#  if config.is_train:
#    for i in xrange(len(data)):
#      input_, label_ = preprocess(data[i], config.scale)
#
#      if len(input_.shape) == 3:
#        h, w, _ = input_.shape
#      else:
#        h, w = input_.shape
#
#      for x in range(0, h-config.image_size+1, config.stride):
#        for y in range(0, w-config.image_size+1, config.stride):
#          sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
#          sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]
#
#          # Make channel value
#          sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
#          sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
#
#          sub_input_sequence.append(sub_input)
#          sub_label_sequence.append(sub_label)
#
#  else:
#    input_, label_ = preprocess(data[2], config.scale)
#
#    if len(input_.shape) == 3:
#      h, w, _ = input_.shape
#    else:
#      h, w = input_.shape
#
#    # Numbers of sub-images in height and width of image are needed to compute merge operation.
#    nx = ny = 0 
#    for x in range(0, h-config.image_size+1, config.stride):
#      nx += 1; ny = 0
#      for y in range(0, w-config.image_size+1, config.stride):
#        ny += 1
#        sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
#        sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]
#        
#        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
#        sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
#
#        sub_input_sequence.append(sub_input)
#        sub_label_sequence.append(sub_label)
#
#  """
#  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
#  (sub_input_sequence[0]).shape : (33, 33, 1)
#  """
#  # Make list to numpy array. With this transform
#  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
#  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]
#
#  make_data(sess, arrdata, arrlabel)
#
#  if not config.is_train:
#    return nx, ny
  
  

