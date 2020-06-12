import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def get_data(key):
	store = pd.HDFStore('curve.h5','r')
	data = store.get(key)
	store.close()
	return data.to_numpy()


while True:
	xmin = float(input("please input xmin:"))
	xmax = float(input("please input xmax:"))
	plt.close()
	
	loss = get_data('loss')
	val_loss = get_data('val_loss')
	counter = get_data('count')
	
	loss = loss[int(len(loss)*xmin):int(len(loss)*xmax)]
	val_loss = val_loss[int(len(val_loss)*xmin):int(len(val_loss)*xmax)]
	counter = counter[int(len(counter)*xmin):int(len(counter)*xmax)]
	
	plt.title("loss & val_loss")
	plt.plot(counter,loss,color = 'r')
	plt.plot(counter,val_loss,color = 'b')
	plt.show()