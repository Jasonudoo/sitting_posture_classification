import glob
import pickle

test_features = []
test_labels = []
train_features = []
train_labels = []
features = []
labels = []
maxcount = 288
class_num = 8
feature_num = 14
testName = 'ginger'
folderName = ['1_proper','3_lying','4_left','5_right','6_leftcross','7_rightcross','8_leftcross1','9_rightcross1']
for i in folderName:
	file_list = glob.glob('data0609/'+i+"/*.txt")
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
			if testName in j:
				test_features.append(list(map(int,line)))
				test_labels.append(folderName.index(i))
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
with open('data.cpickle','wb') as fw:
	pickle.dump([train_features,train_labels,test_features,test_labels],fw)

# data = pickle.load(open('data.cpickle','rb'))
# import pdb;pdb.set_trace()
