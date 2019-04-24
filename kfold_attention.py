from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.merge import multiply
from keras.layers import Input, Dense
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from PIL import Image, ImageDraw
import sys
import pickle
import glob

test_features = []
test_labels = []
train_features = []
train_labels = []
features = []
labels = []
maxcount = 288
class_num = 8
feature_num = 14
ignore_back = False
training = False
data_folder = 'data0609'
folderName = ['1_proper','3_lying','4_left','5_right','6_leftcross','7_rightcross','8_leftcross1','9_rightcross1']
# testName = ['andy_','chiang_','chris_','cliff_','eric_','eric2_','ethan_','ginger_','howard_','jessica_','lulu_','morris2_','nemo_','nemo2_','ruby_','ryan_','ryan2_','sara_','scott_','weiting_','wen_','yao2_','yuwen_']
testName = glob.glob(data_folder+'/'+folderName[0]+'/*.txt')
fresult = open('result.txt','w')
fresult.close()
for i in range(len(testName)):
	string = folderName[0].replace('1_','')
	testName[i] = testName[i].replace(data_folder+'/'+folderName[0]+'/','').replace(string+'.txt','')
# removeName = ['chiang_','cliff_','chunhao_','ryan_']
# removeName = ['chiang_','chris_','cliff_','eric_','hhvs1354_','chunhao_','sara_','ryan_']
# for i in range(len(removeName)):
# 	testName.remove(removeName[i])

for name in testName:
	test_features.append([])
	test_labels.append([])
	train_features.append([])
	train_labels.append([])
	for i in folderName:
		file_list = glob.glob(data_folder+'/'+i+"/*.txt")
		for j in file_list:
			f = open(j,'r')
			count = 0
			for k in f:
				if count == maxcount:
					break
				line = k.split()
				if len(line) != feature_num:
					continue
				temp = list(map(int,line))
				# if ignore_back:
				# 	temp = temp[0:7]+temp[14:21]
				if name in j:
					test_features[-1].append(temp)
					test_labels[-1].append(folderName.index(i))
				else:
					train_features[-1].append(temp)
					train_labels[-1].append(folderName.index(i))
				count += 1


accuracy_list = []
acc_num_people = []
for i in range(class_num):
	acc_num_people.append([0]*10)

average_feature = []
for i in range(class_num):
	average_feature.append([0]*feature_num)

for val in range(len(testName)):

	batch_size = 128
	nb_classes = class_num
	nb_epoch = 100

	# the data, shuffled and split between train and test sets
	[X_train, y_train, X_test, y_test] = [train_features[val],train_labels[val],test_features[val],test_labels[val]]
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

	if training == True:
		inputs = Input(shape=(feature_num,))
		attention_hidden = Dense(64)(inputs)
		attention = Dense(feature_num, activation='softmax')(attention_hidden)
		new_inputs = multiply([attention,inputs])

		hidden = Dense(128, activation='tanh')(new_inputs)
		hidden = Dropout(0.2)(hidden)
		predictions = Dense(nb_classes, activation='softmax')(hidden)

		model = Model(inputs=inputs, outputs=predictions)

		model.summary()

		model.compile(loss='categorical_crossentropy',
		              optimizer=SGD(lr=0.01),
		              metrics=['accuracy'])
		checkpointer = ModelCheckpoint(filepath='kfold_attention_model/'+testName[val]+'.h5', verbose=0, save_best_only=True)
		history = model.fit(X_train, Y_train,
		                    batch_size=batch_size, nb_epoch=nb_epoch,
		                    verbose=1, validation_data=(X_test, Y_test),
		                    callbacks=[checkpointer])
	# model.save('kfold_model/'+testName[val]+'.h5')

	# load model
	model = load_model('kfold_attention_model/'+testName[val]+'.h5')

	# evaluate
	score = model.evaluate(X_test, Y_test, verbose=0)
	error_matrix = np.zeros((nb_classes,nb_classes))
	results = model.predict(X_test)
	for i in range(len(results)):
		error_matrix[list(Y_test[i]).index(max(Y_test[i]))][list(results[i]).index(max(results[i]))]+=1
	print(error_matrix)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	accuracy_list.append(score[1])


	fresult = open('result.txt','a')
	fresult.write(testName[val]+'- error matrix: \n')
	for error in error_matrix:
		fresult.write(str(error)+'\n')
	for i in range(len(error_matrix)):
		each_pose_acc = float(error_matrix[i][i])/sum(error_matrix[i])
		fresult.write(str(each_pose_acc)+'\n')
		try:
			acc_num_people[i][int(each_pose_acc*10-0.1)] += 1
		except:
			import pdb;pdb.set_trace()
	fresult.write(testName[val]+'- accuracy: '+str(accuracy_list[val])+'\n')
	fresult.close()


	inp = model.input                                           # input placeholder
	outputs = [layer.output for layer in model.layers]          # all layer outputs
	functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

	# Testing
	layer_outs = [func([X_test, 1.]) for func in functors]

	results = model.predict(X_test)

	for f in range(nb_classes):
		average = [0]*feature_num
		for i in range(maxcount):
			average += layer_outs[2][0][f*maxcount+i]
		average = average/maxcount
		average_feature[f] = average_feature[f]+average


fresult = open('result.txt','a')
fresult.write('average accuracy: '+str(sum(accuracy_list)/len(testName))+'\n')
for pose in acc_num_people:
	fresult.write(str(pose)+'\n')
fresult.close()
# with open('features.pickle','w') as f:
# 	pickle.dump(f,average_feature)
# draw
for f in range(nb_classes):
	average_feature[f] = average_feature[f]/sum(average_feature[f])


	
	im = Image.new('RGB', (640, 640), (255, 255, 255))
	draw = ImageDraw.Draw(im)
	draw.text ((300, 10), 'Table', fill=(0,0,0), font=None)
	for i in range(7):
		draw.ellipse((390,100+i*60, 440, 150+i*60), fill = (255, int(255-255*average_feature[f][i]), int(255-255*average_feature[f][i])))
	for i in range(7):
		draw.ellipse((200,100+i*60, 250, 150+i*60), fill = (255, int(255-255*average_feature[f][13-i]), int(255-255*average_feature[f][13-i])))
	del draw
	im.save('attention/'+folderName[f]+'.png', "PNG")


im = Image.new('RGB', (640, 640), (255, 255, 255))
draw = ImageDraw.Draw(im)
draw.text ((300, 10), 'Table', fill=(0,0,0), font=None)
for i in range(7):
	draw.ellipse((390,100+i*60, 440, 150+i*60), fill = (255, int(255-255*average_feature[f][i]), int(255-255*average_feature[f][i])))
for i in range(7):
	draw.ellipse((200,100+i*60, 250, 150+i*60), fill = (255, int(255-255*average_feature[f][13-i]), int(255-255*average_feature[f][13-i])))
del draw
im.save('attention/'+'all_feature'+'.png', "PNG")