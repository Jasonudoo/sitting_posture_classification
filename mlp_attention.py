'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

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
import pickle

batch_size = 128
nb_classes = 8
nb_epoch = 200
nb_featres = 14
# the data, shuffled and split between train and test sets
[X_train, y_train, X_test, y_test] = pickle.load(open('data.cpickle','rb'))
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


inputs = Input(shape=(nb_featres,))
attention_hidden = Dense(64)(inputs)
attention = Dense(nb_featres, activation='softmax')(attention_hidden)
new_inputs = multiply([attention,inputs])

hidden = Dense(128, activation='tanh')(new_inputs)
hidden = Dropout(0.2)(hidden)
predictions = Dense(nb_classes, activation='softmax')(hidden)

model = Model(inputs=inputs, outputs=predictions)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.h5', verbose=0, save_best_only=True)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[checkpointer])
# model.save('model.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
error_matrix = np.zeros((nb_classes,nb_classes))
results = model.predict(X_test)
for i in range(len(results)):
	error_matrix[list(Y_test[i]).index(max(Y_test[i]))][list(results[i]).index(max(results[i]))]+=1
print(error_matrix)
print('Test score:', score[0])
print('Test accuracy:', score[1])
# import pdb;pdb.set_trace()