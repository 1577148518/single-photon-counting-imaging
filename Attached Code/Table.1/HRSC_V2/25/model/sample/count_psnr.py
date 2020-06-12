import os
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage

import numpy as np
import math

import glob	
    
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
      
def get_image():
	img_orignal = imread("./orignal.png", is_grayscale=True)
	img_rebuilt = imread("./test_image.png", is_grayscale=True)
	if len(img_rebuilt.shape) == 3:
		h, w, _ = img_rebuilt.shape
		img_orignal = img_orignal[0:h, 0:w, :]
	else:
		h, w = img_rebuilt.shape
		img_orignal = img_orignal[0:h, 0:w]
	return [img_orignal,img_rebuilt]
   

def imsave(image, path):
    return scipy.misc.imsave(path, image)
    
def psnr_counter(img1, img2):
	mse = np.mean( (img1/255. - img2/255.) ** 2 )
	if mse < 1.0e-10:
		return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
img = get_image()
imsave(img[0],"./oringal__0.png")
imsave(img[1],"./rebuilt__0.png")
print(psnr_counter(img[0], img[1]))

	