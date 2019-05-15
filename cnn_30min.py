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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pickle
import glob
import os

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
dataset = 'data190512_30min'
folderName = ['proper','lying','left','right','leftcross','rightcross','leftcross1','rightcross1']
if int(step) <= 1:
	# the data, shuffled and split between train and test sets
	[X_train, y_train, X_test, y_test] = pickle.load(open(dataset+'/data/ann_30min.cpickle','rb'))
	for i in range(len(X_train)):
		X_train[i] = [X_train[i][-1]]+X_train[i]+[X_train[i][0]]
		for j in range(len(X_train[i])):
			X_train[i][j] = [X_train[i][j]]
	for i in range(len(X_test)):
		X_test[i] = [X_test[i][-1]]+X_test[i]+[X_test[i][0]]
		for j in range(len(X_test[i])):
			X_test[i][j] = [X_test[i][j]]		
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
	model.add(Convolution1D(nb_filter=256,
                            filter_length=3,
                            border_mode='valid',
                            activation='relu', 
                            input_shape=(feature_num+2,1)))
	model.add(MaxPooling1D(pool_length=2))
	#model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.summary()

	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	checkpointer = ModelCheckpoint(filepath=dataset+'/model_cnn_256_3.h5', verbose=0, save_best_only=True)
	history = model.fit(X_train, Y_train,
	                    batch_size=batch_size, nb_epoch=nb_epoch,
	                    verbose=1, validation_data=(X_test, Y_test),
	                    callbacks=[checkpointer])
	model.save(dataset+'/model_cnn_256_3.h5')


if int(step) <= 2:
	result_name = 'result_cnn_256_3'
	result_file = open(dataset+'/'+result_name+'.txt','w')
	result_folder = result_name
	if not os.path.exists(dataset+'/'+result_folder):
		os.mkdir(dataset+'/'+result_folder)	
	model = load_model(dataset+'/model_cnn_256_3.h5')
	file_list = glob.glob(dataset+"/data/*.cpickle")
	pose_accuracy = [0]*nb_classes
	pose_all = [0]*nb_classes
	people_accuracy = []
	all_smooth_accuracy = []
	for i in file_list:
		print(i, file = result_file)
		[X_train, y_train, X_test, y_test] = pickle.load(open(i,'rb'))
		for j in range(len(X_train)):
			X_train[j] = [X_train[j][-1]]+X_train[j]+[X_train[j][0]]
			for k in range(len(X_train[j])):
				X_train[j][k] = [X_train[j][k]]
		for j in range(len(X_test)):
			X_test[j] = [X_test[j][-1]]+X_test[j]+[X_test[j][0]]
			for k in range(len(X_test[j])):
				X_test[j][k] = [X_test[j][k]]			
		X_test = np.array(X_test)
		y_test = np.array(y_test)
		X_test = X_test.astype('float32')
		X_test /= 1024
		Y_test = np_utils.to_categorical(y_test, nb_classes)
		print(X_test.shape[0], 'test samples', file = result_file)
		score = model.evaluate(X_test, Y_test, verbose=0)
		error_matrix = np.zeros((nb_classes,nb_classes))
		results_probability = model.predict(X_test)
		for j in range(len(results_probability)):
			error_matrix[list(Y_test[j]).index(max(Y_test[j]))][list(results_probability[j]).index(max(results_probability[j]))]+=1
		print(error_matrix, file = result_file)
		for j in range(len(error_matrix)):
			pose_all[j] += sum(error_matrix[j])
			pose_accuracy[j] += error_matrix[j][j]
		print('Test score:', score[0], file = result_file)
		print('Test accuracy:', score[1], file = result_file)
		people_accuracy.append(score[1])

		results_class = []
		for j in range(len(results_probability)):
			results_class.append(list(results_probability[j]).index(max(results_probability[j])))
		# import pdb;pdb.set_trace()
		plt.cla()
		plt.axis([0, len(results_class), -1, nb_classes])
		plt.rcParams['ytick.labelsize'] = 'small'
		plt.yticks(np.arange(-1,nb_classes), ([' ']+folderName+[' ']))
		plt.plot(range(len(results_class)), results_class,'ro',markersize=1, label="predict result")
		plt.plot(range(len(y_test)), list(np.array(y_test)+0.2),'bo',markersize=1, label="target")
		plt.legend(loc='upper right')
		plt.show()
		plt.savefig(i.replace('.cpickle','.true.png').replace('/data/','/'+result_folder+'/'))
		
		# smooth
		results_class_smmoth = results_class[:]
		width = 9 #51
		threashold = width/2+1
		half_window = width/2
		for j in range(1,len(results_class_smmoth)-width):
			if results_class_smmoth[j]==results_class_smmoth[j-1]: # find the change point
				continue
			count = 0
			window = []
			for k in range(-half_window,half_window+1):
				window.append(results_class_smmoth[j+k])
				if results_class_smmoth[j+k]==results_class_smmoth[j]:
					count += 1
			if count < threashold:
				results_class_smmoth[j] = max(set(window), key = window.count)


		accuracy = 0.0
		for j in range(len(results_class_smmoth)):
			if results_class_smmoth[j]==y_test[j]:
				accuracy+=1
		accuracy /= len(results_class_smmoth)
		all_smooth_accuracy.append(accuracy)
		print('Smooth accuracy: '+str(accuracy)+'\n', file = result_file)
		plt.cla()
		plt.axis([0, len(results_class), -1, nb_classes])
		plt.rcParams['ytick.labelsize'] = 'small'
		plt.yticks(np.arange(-1,nb_classes), ([' ']+folderName+[' ']))
		plt.plot(range(len(results_class_smmoth)), results_class_smmoth,'ro',markersize=1, label="predict result")
		plt.plot(range(len(y_test)), list(np.array(y_test)+0.2),'bo',markersize=1, label="target")
		plt.legend(loc='upper right')
		plt.show()
		plt.savefig(i.replace('.cpickle','.smooth.png').replace('/data/','/'+result_folder+'/'))
		# import pdb;pdb.set_trace()
	pose_accuracy = np.array(pose_accuracy)/np.array(pose_all)
	print('pose_accuracy = ', file = result_file)
	print(pose_accuracy, file = result_file)
	print(sum(pose_accuracy)/nb_classes, file = result_file)

	print('\npeople average accuracy: '+str(sum(people_accuracy)/len(file_list)), file = result_file)
	print('\nsmooth accuracy: '+str(sum(all_smooth_accuracy)/len(file_list)), file = result_file)