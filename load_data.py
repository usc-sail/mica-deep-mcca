# this is specifically written for the three noisy MNIST case
# might not generalize!

import os
import numpy as np
import theano
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from collections import Counter

# assume these files exist in the same location as this script and all others
mat_filenames = ['mnist-with-awgn.mat', 'mnist-with-motion-blur.mat',
				'mnist-with-reduced-contrast-and-awgn.mat']
def load_noisy_mnist_data(mat_filenames=mat_filenames):
	"""
	matfiles have train_x,y and test_x,y need to generate validation set
	"""
	data_list = []
	for mat_file_ in mat_filenames:
		m_dict = loadmat(mat_file_)
		all_train_x = m_dict['train_x']
		all_train_y = m_dict['train_y']
		# the train set has 60K so use 10K as val
		train_x, val_x, train_y, val_y = train_test_split(all_train_x, all_train_y, 
								test_size = (1./6), random_state=27)
		# onehot to interger for {}_y
		train_set = make_numpy_array(train_x, np.argmax(train_y, axis=1))
		val_set = make_numpy_array(val_x, np.argmax(val_y, axis=1))
		test_set = make_numpy_array(m_dict['test_x'], np.argmax(m_dict['test_y'], axis=1))

		data_list.append([train_set, val_set, test_set])
	return data_list

def make_numpy_array(data_x, data_y):
	"""converts the input to numpy arrays"""
	data_x = np.asarray(data_x, dtype=theano.config.floatX)
	data_y = np.asarray(data_y, dtype='int32')
	return (data_x, data_y)
