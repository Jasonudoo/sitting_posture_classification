import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.utils import np_utils
import pickle
from PIL import Image, ImageDraw
import sys

folderName = ['1_proper','3_lying','4_left','5_right','6_leftcross','7_rightcross','8_leftcross1','9_rightcross1']
batch_size = 128
nb_classes = 8
nb_epoch = 200
nb_featres = 14
maxcount = 288

[X_train, y_train, X_test, y_test] = pickle.load(open('data.cpickle','rb'))
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 1024
X_test /= 1024
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = load_model('model.h5')


inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

# Testing
layer_outs = [func([X_test, 1.]) for func in functors]

results = model.predict(X_test)

error_matrix = np.zeros((nb_classes,nb_classes))
for i in range(len(results)):
	error_matrix[list(Y_test[i]).index(max(Y_test[i]))][list(results[i]).index(max(results[i]))]+=1
print(error_matrix)

# draw
for f in range(nb_classes):
	average = [0]*nb_featres
	for i in range(maxcount):
		average += layer_outs[2][0][f*maxcount+i]
	average = average/maxcount
	im = Image.new('RGB', (640, 640), (255, 255, 255))
	draw = ImageDraw.Draw(im)
	draw.text ((300, 10), 'Table', fill=(0,0,0), font=None)
	for i in range(7):
		draw.ellipse((390,100+i*60, 440, 150+i*60), fill = (255*average[i], 0, 0))
	for i in range(7):
		draw.ellipse((200,100+i*60, 250, 150+i*60), fill = (255*average[13-i], 0, 0))
	del draw
	im.save('attention/'+folderName[f]+'.png', "PNG")

import pdb;pdb.set_trace()