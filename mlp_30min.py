'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import load_model
import pickle
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
if len(sys.argv)==1:
	step = 1
else:
	step = sys.argv[1]
batch_size = 128
nb_classes = 8
feature_num = 14
nb_epoch = 100
dataset = 'data0723'
folderName = ['proper','lying','left','right','leftcross','rightcross','leftcross1','rightcross1']
if int(step) <= 1:
	# the data, shuffled and split between train and test sets
	[X_train, y_train, X_test, y_test] = pickle.load(open(dataset+'/ginger_30min.cpickle','rb'))
	X_train = np.array(X_train)
	y_train = np.array(y_train)
	X_test = np.array(X_test)
	y_test = np.array(y_test)
	#X_train = X_train.reshape(60000, 784)
	#X_test = X_test.reshape(10000, 784)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 1024
	X_test /= 1024
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	model = Sequential()
	model.add(Dense(128, input_shape=(feature_num,)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(Activation('tanh'))
	model.add(Dropout(0.2))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
	              optimizer=SGD(lr=0.01),
	              metrics=['accuracy'])

	history = model.fit(X_train, Y_train,
	                    batch_size=batch_size, nb_epoch=nb_epoch,
	                    verbose=1, validation_data=(X_test, Y_test))
	model.save(dataset+'/model.h5')


if int(step) <= 2:
	model = load_model(dataset+'/model.h5')
	file_list = glob.glob(dataset+"/*.cpickle")
	pose_accuracy = [0]*nb_classes
	pose_all = [0]*nb_classes
	people_accuracy = []
	for i in file_list:
		print(i)
		[X_train, y_train, X_test, y_test] = pickle.load(open(i,'rb'))
		X_test = np.array(X_test)
		y_test = np.array(y_test)
		X_test = X_test.astype('float32')
		X_test /= 1024
		Y_test = np_utils.to_categorical(y_test, nb_classes)
		print(X_test.shape[0], 'test samples')
		score = model.evaluate(X_test, Y_test, verbose=0)
		error_matrix = np.zeros((nb_classes,nb_classes))
		results_probability = model.predict(X_test)
		for j in range(len(results_probability)):
			error_matrix[list(Y_test[j]).index(max(Y_test[j]))][list(results_probability[j]).index(max(results_probability[j]))]+=1
		print(error_matrix)
		for j in range(len(error_matrix)):
			pose_all[j] += sum(error_matrix[j])
			pose_accuracy[j] += error_matrix[j][j]
		print('Test score:', score[0])
		print('Test accuracy:', score[1])
		people_accuracy.append(score[1])

		results_class = []
		for j in range(len(results_probability)):
			results_class.append(list(results_probability[j]).index(max(results_probability[j])))
		# import pdb;pdb.set_trace()
		plt.cla()
		plt.axis([0, len(results_class), -1, nb_classes])
		plt.rcParams['ytick.labelsize'] = 'small'
		plt.yticks(np.arange(-1,nb_classes), ([' ']+folderName+[' ']))
		plt.plot(range(len(results_class)), results_class,'ro',markersize=3, label="predict result")
		plt.plot(range(len(y_test)), list(np.array(y_test)+0.2),'bo',markersize=3, label="target")
		plt.legend(loc='upper right')
		plt.show()
		plt.savefig(i.replace('.cpickle','.true.png'))
		
		# smooth
		results_class_smmoth = results_class[:]
		width = 5
		min_mode = width*0.4
		for Iter in range(3):
			for j in range(width,len(results_class_smmoth)-width):
				window = []
				for k in range(j-width, j):
					window.append(results_class_smmoth[k])
				mode_num_left = max(set(window), key=window.count)
				if window.count(mode_num_left)<min_mode:
					continue

				window = []
				for k in range(j, j+width):
					window.append(results_class_smmoth[k])
				mode_num_right = max(set(window), key=window.count)
				if window.count(mode_num_right)<min_mode:
					continue

				if mode_num_left==mode_num_right:
					results_class_smmoth[j] = mode_num_left
		
		accuracy = 0.0
		for j in range(len(results_class_smmoth)):
			if results_class_smmoth[j]==y_test[j]:
				accuracy+=1
		accuracy /= len(results_class_smmoth)
		print('Smooth accuracy: '+str(accuracy))
		print('\n')
		plt.cla()
		plt.plot(range(len(results_class_smmoth)), results_class_smmoth,'ro',markersize=1)
		plt.show()
		plt.savefig(i.replace('cpickle','smooth.png'))
		# import pdb;pdb.set_trace()
	pose_accuracy = np.array(pose_accuracy)/np.array(pose_all)
	print('pose_accuracy = ')
	print(pose_accuracy)
	print(sum(pose_accuracy)/nb_classes)

	print('\npeople average accuracy: '+str(sum(people_accuracy)/len(file_list)))