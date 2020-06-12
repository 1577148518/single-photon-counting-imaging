# -*- coding=utf-8 -*-
import tensorflow as tf
from tools import make_dir
from tools import input_setup

from backward import backward

flags = tf.app.flags
flags.DEFINE_integer("epoch",         10000,   "Number of epoch [15000]")
flags.DEFINE_integer("batch_size",    128,     "The size of batch images [128]")
flags.DEFINE_integer("image_size",    32,      "The size of image to use [28]")
flags.DEFINE_integer("label_size",    32,      "The size of label to produce [28]")
flags.DEFINE_integer("compress_size", 256,	   "The size of compress matrix to produce [14]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_float("learning_rate",   0.00001,   "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("learding_rate_delay",1,"The learning rate decay")
flags.DEFINE_float("REGULARIZER",0.0001,"REGULARIZER")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 10, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

def main(_):
	print("start.......")
	
	make_dir()
	
	with tf.Session() as sess:
		print("compress rate is:",FLAGS.compress_size / FLAGS.image_size/FLAGS.image_size)
		if FLAGS.is_train:
			input_setup(FLAGS)				#制作数据集
			nx, ny = 0 , 0
		else:
			nx, ny = input_setup(FLAGS)
	
	backward(nx,ny)
		
	
if __name__ == '__main__':
	tf.app.run()