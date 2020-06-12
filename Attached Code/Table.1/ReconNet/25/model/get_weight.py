from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import scipy.io as sio
import pickle
import numpy as np

print("starting......")
# 读取存储在checkpoint里的各层权重
model_dir = '.\\checkpoint\\Autoencoder_32\\Autoencoder.model-75600'
reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
var_to_shape_map = reader.get_variable_to_shape_map()

pkl_dict = {}      #创建一个字典
for key in sorted(var_to_shape_map):
    pkl_dict[key] = reader.get_tensor(key) # numpy.ndarray

print("Variable is:",var_to_shape_map)
output = open('./data/data_0.02.pkl', 'wb')   #写入文件 
pickle.dump(pkl_dict, output)
output.close()

pkl_file = open('./data/data_0.02.pkl', 'rb')
data = pickle.load(pkl_file) 

fcw_1 = data['Variable']

print(fcw_1)
print(fcw_1.shape)
sio.savemat("./data/data_0.02.mat", {"fcw_1": fcw_1})#储存为.mat格式