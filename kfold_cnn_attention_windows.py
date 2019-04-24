from __future__ import print_function
import numpy as np
# np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda, Layer, Reshape
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.merge import multiply, concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from PIL import Image, ImageDraw
import pickle
import glob
import os
from keras.utils import plot_model

test_features = []
test_features2 = []
test_labels = []
train_features = []
train_features2 = []
train_labels = []
features = []
labels = []
training = True
maxcount = 260
class_num = 8
feature_num = 14 
ignore_back = True
batch_size = 128
nb_epoch = 100
data_type = 'butt' #databack0628 is butt_back
data_folder = 'data0609/'
folderName = ['1_proper','3_lying','4_left','5_right','6_leftcross','7_rightcross','8_leftcross1','9_rightcross1']
# testName = ['andy_','chiang_','chris_','cliff_','eric_','eric2_','ethan_','ginger_','howard_','jessica_','lulu_','morris2_','nemo_','nemo2_','ruby_','ryan_','ryan2_','sara_','scott_','weiting_','wen_','yao2_','yuwen_']
testName = glob.glob(data_folder+folderName[0]+'/*.txt')
fresult = open(data_folder+'result_cnn_attention.txt','w')
fresult.close()
for i in range(len(testName)):
    string = folderName[0].replace('1_','')
    testName[i] = testName[i].replace(data_folder+folderName[0]+'\\','').replace(string+'.txt','')
# removeName = ['chiang_','cliff_','chunhao_','ryan_']
# removeName = ['chiang_','chris_','cliff_','eric_','hhvs1354_','chunhao_','sara_','ryan_']
# for i in range(len(removeName)):
#     testName.remove(removeName[i])

for name in testName:
    test_features.append([])
    test_features2.append([])
    test_labels.append([])
    train_features.append([])
    train_features2.append([])
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
                temp2 = [temp[-1]]+temp+[temp[0]]
                for l in range(len(temp2)):
                    temp2[l] = [temp2[l]]
                if name in j:
                    test_features[-1].append(temp)
                    test_features2[-1].append(temp2)
                    test_labels[-1].append(folderName.index(i))
                else:
                    train_features[-1].append(temp)
                    train_features2[-1].append(temp2)
                    train_labels[-1].append(folderName.index(i))
                count += 1

if not os.path.exists(data_folder+'kfold_model_cnn_attention/'):
    os.mkdir(data_folder+'kfold_model_cnn_attention/')

accuracy_list = [0]*len(testName)
acc_num_people = []
for i in range(class_num):
    acc_num_people.append([0]*10)

average_feature = []
for i in range(class_num):
    average_feature.append([0]*feature_num)

for val in range(len(testName)):

    nb_classes = class_num

    # the data, shuffled and split between train and test sets
    [X_train, y_train, X_test, y_test] = [train_features[val],train_labels[val],test_features[val],test_labels[val]]
    [X_train2, X_test2] = [train_features2[val],test_features2[val]]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_train2 = np.array(X_train2)
    X_test2 = np.array(X_test2)
    #X_train = X_train.reshape(60000, 784)
    #X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 1024
    X_test /= 1024
    X_train2 = X_train2.astype('float32')
    X_test2 = X_test2.astype('float32')
    X_train2 /= 1024
    X_test2 /= 1024    
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    if training:

        inputs1 = Input(shape=(feature_num,))
        inputs2 = Input(shape=(feature_num+2,1))
        attention_hidden = Dense(64)(inputs1)
        attention = Dense(feature_num, activation='softmax', name="attention")(attention_hidden)
        conv = Convolution1D(nb_filter=5,
                             filter_length=3,
                             border_mode='valid',
                             activation='relu', 
                             input_shape=(feature_num+2,1))(inputs2)
        attention = Reshape((14,1))(attention)
        new_inputs = multiply([conv,attention])
        pool = MaxPooling1D(pool_length=2)(new_inputs)
        flat = Flatten()(pool)
        predictions = Dense(nb_classes, activation='softmax')(flat)

        model = Model(inputs=[inputs1,inputs2], outputs=predictions)

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath=data_folder+'kfold_model_cnn_attention/'+testName[val]+'.h5', verbose=0, save_best_only=True)
        #import pdb;pdb.set_trace()
        history = model.fit([X_train,X_train2], Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=1, validation_data=([X_test,X_test2], Y_test),
                            callbacks=[checkpointer])
        # model.save('kfold_model_cnn_attention/'+testName[val]+'.h5')

    # load model
    model = load_model(data_folder+'kfold_model_cnn_attention/'+testName[val]+'.h5')

    plot_model(model, to_file=testName[val]+'.png')

    # evaluate
    score = model.evaluate([X_test,X_test2], Y_test, verbose=0)
    error_matrix = np.zeros((nb_classes,nb_classes))
    results = model.predict([X_test,X_test2])
    for i in range(len(results)):
        error_matrix[list(Y_test[i]).index(max(Y_test[i]))][list(results[i]).index(max(results[i]))]+=1
    print(error_matrix)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    accuracy_list[val] = score[1]


    fresult = open(data_folder+'result_cnn_attention.txt','a')
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

    # calculate attention weight
    #import pdb;pdb.set_trace()
    layer_name = 'attention'
    attention_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    attention_output = attention_layer_model.predict([X_test,X_test2])
    
    for f in range(nb_classes):
        average = [0]*feature_num
        for i in range(maxcount):
            average += attention_output[f*maxcount+i]
        average = average/maxcount
        average_feature[f] = average_feature[f]+average

if not os.path.exists(data_folder+'cnn_attention_result/'):
    os.mkdir(data_folder+'cnn_attention_result/')
fresult = open(data_folder+'result_cnn_attention.txt','a')
fresult.write('average accuracy: '+str(sum(accuracy_list)/len(testName))+'\n')
for pose in acc_num_people:
    fresult.write(str(pose)+'\n')
fresult.close()
# with open('features.pickle','w') as f:
#     pickle.dump(f,average_feature)
# draw
all_feature = [0]*feature_num
for f in range(nb_classes):
    average_feature[f] = average_feature[f]/sum(average_feature[f])
    all_feature += average_feature[f]
    im = Image.new('RGB', (640, 640), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text ((300, 10), 'Table', fill=(0,0,0), font=None)
    for i in range(7):
        draw.ellipse((390,100+i*60, 440, 150+i*60), fill = (255, int(255-255*average_feature[f][i]), int(255-255*average_feature[f][i])))
        draw.text((390, 100+i*60),str(round(average_feature[f][i],3)),(0,0,0),font=None)
    for i in range(7):
        draw.ellipse((200,100+i*60, 250, 150+i*60), fill = (255, int(255-255*average_feature[f][13-i]), int(255-255*average_feature[f][13-i])))
        draw.text((200, 100+i*60),str(round(average_feature[f][13-i],3)),(0,0,0),font=None)
    del draw
    im.save(data_folder+'cnn_attention_result/'+folderName[f]+'.png', "PNG")
all_feature/=nb_classes

im = Image.new('RGB', (640, 640), (255, 255, 255))
draw = ImageDraw.Draw(im)
draw.text ((300, 10), 'Table', fill=(0,0,0), font=None)
for i in range(7):
    draw.ellipse((390,100+i*60, 440, 150+i*60), fill = (255, int(255-255*all_feature[i]), int(255-255*all_feature[i])))
    draw.text((390, 100+i*60),str(round(all_feature[i],3)),(0,0,0),font=None)
for i in range(7):
    draw.ellipse((200,100+i*60, 250, 150+i*60), fill = (255, int(255-255*all_feature[13-i]), int(255-255*all_feature[13-i])))
    draw.text((200, 100+i*60),str(round(all_feature[13-i],3)),(0,0,0),font=None)
del draw
im.save(data_folder+'cnn_attention_result/'+'all_feature'+'.png', "PNG")
