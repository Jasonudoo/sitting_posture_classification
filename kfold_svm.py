from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from libsvm.python.svmutil import *
import pickle
import glob
import os, sys
test_features = []
test_labels = []
train_features = []
train_labels = []
features = []
labels = []
maxcount = 260
class_num = 8
feature_num = 14
ignore_back = False
data_type = 'butt' #databack0628 is butt_back
data_folder = 'data0609/'
model_folder = 'kfold_model_svm_radialBasis/'
result_file = 'result_svm_radialBasis.txt'
training = True
folderName = ['1_proper','3_lying','4_left','5_right','6_leftcross','7_rightcross','8_leftcross1','9_rightcross1']
# testName = ['andy_','chiang_','chris_','cliff_','eric_','eric2_','ethan_','ginger_','howard_','jessica_','lulu_','morris2_','nemo_','nemo2_','ruby_','ryan_','ryan2_','sara_','scott_','weiting_','wen_','yao2_','yuwen_']
testName = glob.glob(data_folder+'/'+folderName[0]+'/*.txt')
fresult = open(data_folder+result_file,'w')
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
		file_list = glob.glob(data_folder+i+"/*.txt")
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
				if  data_type=='butt_back' and ignore_back:
					temp = temp[0:7]+temp[14:21]
				if '/'+name in j:
					test_features[-1].append(temp)
					test_labels[-1].append(folderName.index(i))
				else:
					train_features[-1].append(temp)
					train_labels[-1].append(folderName.index(i))
				count += 1


accuracy_list = []
acc_num_people = []
all_error_matrix = np.zeros((class_num,class_num))
for i in range(class_num):
	acc_num_people.append([0]*10)
if not os.path.exists(data_folder+model_folder):
	os.mkdir(data_folder+model_folder)
for val in range(len(testName)):
	if not os.path.exists(data_folder+model_folder+testName[val]):
		os.mkdir(data_folder+model_folder+testName[val])
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
	X_train = X_train.tolist()
	
	if training:
		classIndexInTrain = []
		model = [0]*class_num
		for i in range(class_num):
			classIndexInTrain.append(y_train.tolist().index(i))
			model[i] = [0]*class_num
		classIndexInTrain.append(len(y_train))
		intervel = classIndexInTrain[1]-classIndexInTrain[0]
		for i in range(class_num):
			for j in range(i+1,class_num):
				model[i][j] = svm_train([1]*intervel+[-1]*intervel, X_train[classIndexInTrain[i]:classIndexInTrain[i+1]]+X_train[classIndexInTrain[j]:classIndexInTrain[j+1]], '-c 4 -t 2')
				svm_save_model(data_folder+model_folder+testName[val]+'/'+str(i)+str(j)+'.model', model[i][j])
	
	
	# model.save('kfold_model/'+testName[val]+'.h5')

	# load model
	model = [0]*class_num
	for i in range(class_num):
		model[i] = [0]*class_num
	for i in range(class_num):
		for j in range(i+1,class_num):
			model[i][j] = svm_load_model(data_folder+model_folder+testName[val]+'/'+str(i)+str(j)+'.model')
			

	# evaluate
	error_matrix = np.zeros((nb_classes,nb_classes))
	label = [0]*len(y_test)
	for i in range(len(y_test)):
		label[i] = [0]*class_num
	for i in range(class_num):
		for j in range(i+1,class_num):
			p_label, p_acc, p_val = svm_predict(y_test, X_test, model[i][j])
			for k in range(len(p_label)):
				if p_label[k]==1:
					label[k][i]+=1
				else:
					label[k][j]+=1
	for i in range(len(label)):
		label[i] = label[i].index(max(label[i]))
	
	#import pdb;pdb.set_trace()
	accuracy = 0
	for i in range(class_num):
		for j in range(maxcount):
			error_matrix[i][label[maxcount*i+j]]+=1
			if label[maxcount*i+j] == i:
				accuracy+=1
	accuracy /= float(len(label))
	print(error_matrix)
	#print('Test score:', score[0])
	print('Test accuracy:', accuracy)
	accuracy_list.append(accuracy)


	fresult = open(data_folder+result_file,'a')
	fresult.write(testName[val]+'- error matrix: \n')
	all_error_matrix += error_matrix	
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



fresult = open(data_folder+result_file,'a')
fresult.write('average accuracy: '+str(sum(accuracy_list)/len(testName))+'\n')
for pose in acc_num_people:
	fresult.write(str(pose)+'\n')
for i in range(len(all_error_matrix)):
	all_error_matrix[i] = all_error_matrix[i]/float(sum(all_error_matrix[i]))
	fresult.write(str(all_error_matrix[i])+'\n')
fresult.close()
table_file = data_folder+result_file.replace('txt','jpg')
execfile("test_table.py")
