from sklearn.datasets import load_iris
from sklearn import tree
import sklearn
import pickle
import numpy as np
import graphviz 
print('The scikit-learn version is {}.'.format(sklearn.__version__))

folderName = ['1_proper','3_lying','4_left','5_right','6_leftcross','7_rightcross','8_leftcross1','9_rightcross1','10_hunchback']
[X_train, y_train, X_test, y_test] = pickle.load(open('data.cpickle','rb'))
model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)

dot_data = tree.export_graphviz(model, out_file='tree.dot', filled =True,
                     feature_names=range(14))  
graph = graphviz.Source(dot_data)

import pdb;pdb.set_trace()
