import glob
import pickle


train_features = []
train_labels = []
features = []
labels = []
maxcount = 288
class_num = 8
feature_num = 14
dataset = 'data0609'
folderName = ['1_proper','3_lying','4_left','5_right','6_leftcross','7_rightcross','8_leftcross1','9_rightcross1']
for i in folderName:
	file_list = glob.glob(dataset+'/'+i+"/*.txt")
	for j in file_list:
		f = open(j,'r')
		temp_str = ''
		count = 0
		for k in f:
			if count == maxcount:
				break
			line = k.split()
			if len(line) != feature_num:
				continue
			else:
				train_features.append(list(map(int,line)))
				train_labels.append(folderName.index(i))
			

			count += 1
			# if k=='\n' and temp_str!='':
			# 	temp_str = temp_str.split(',')
			# 	temp_str.pop()
			# 	features.append(list(map(int,temp_str)))
			# 	labels.append(folderName.index(i))
			# 	temp_str = ''
			# 	count += 1
			# else:
			# 	temp_str+=k.replace('\n','')
		if count < maxcount:
			print(j+' data no enough only '+ str(count) +' \n')

dataset = 'data0723'
bias = 1000
frequency = 6
skip = 120
file_list = glob.glob(dataset+"/*feature.txt")
for i in file_list:
	test_features = []
	test_labels = []	
	f = open(i,'r')
	for j in f:
		line = j.split()
		if len(line)==0:
			continue
		elif len(line) != feature_num:
			print('warning!: '+i+' features has error')
			continue
		else:
			temp = list(map(int,line))
			temp = temp[7:]+list(reversed(temp[:7]))
			test_features.append(temp)
	f.close()

	f=open(i.replace('feature','label'),'r')
	time_start = []
	for j in f:
		line = j.split()
		if len(line)==2:
			time_start.append(line)
			if int(time_start[-1][0])!=0:
				time_start[-1][0]= int(time_start[-1][0])*frequency+bias
	time_start.append([len(test_features),''])
	for j in range(len(time_start)-1):
		# import pdb;pdb.set_trace()
		label_num = int(time_start[j+1][0])-int(time_start[j][0])

		for k in range(len(folderName)):
			if time_start[j][1]==folderName[k][2:]:
				test_labels += [k]*label_num
				break
	f.close()
	
	test_features = test_features[0:len(test_features):skip]
	test_labels = test_labels[0:len(test_labels):skip]
	if len(test_features)!=len(test_labels):
		import pdb;pdb.set_trace()

	with open(i.replace('.feature.txt','')+'.cpickle','wb') as fw:
		pickle.dump([train_features,train_labels,test_features,test_labels],fw)

# data = pickle.load(open('data.cpickle','rb'))
# import pdb;pdb.set_trace()
