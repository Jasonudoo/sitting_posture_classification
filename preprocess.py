import glob
import pickle


features = []
labels = []
maxcount = 70
folderName = ['1_proper','2_leanforward','3_lying','4_left','5_right','6_leftcross','7_rightcross','8_leftcross1','9_rightcross1']
for i in folderName:
	file_list = glob.glob('data/'+i+"/*.txt")
	for j in file_list:
		f = open(j,'r')
		temp_str = ''
		count = 0
		for k in f:
			k = k.replace('\r','')
			if count == maxcount:
				break
			if k=='\n' and temp_str!='':
				temp_str = temp_str.split(',')
				temp_str.pop()
				features.append(map(int,temp_str))
				labels.append(folderName.index(i))
				temp_str = ''
				count += 1
			else:
				temp_str+=k.replace('\n','')
		if count < maxcount:
			print(j+'\n')

test_num = len(features)/10/9
test_features = []
test_labels = []
train_features = []
train_labels = []
count = 0
cur_label = 0
for i in range(len(features)):
	if features[i]==[]:
		continue
	if count<test_num:
		test_features.append(features[i][1:])
		test_labels.append(labels[i])
	else:
		train_features.append(features[i][1:])
		train_labels.append(labels[i])
	if cur_label!=labels[i]:
		cur_label = labels[i]
		count = 0
	else :
		count+=1

with open('data.cpickle','wb') as fw:
	pickle.dump([train_features,train_labels,test_features,test_labels],fw)

# data = pickle.load(open('data.cpickle','rb'))
import pdb;pdb.set_trace()
