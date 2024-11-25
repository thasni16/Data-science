import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data=load_iris()
x=data.data
y=data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=50,test_size=0.25)

classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print('accuracy on train data using gini:',accuracy_score(y_train,classifier.predict(x_train)))
print('accuracy on test data using gini:',accuracy_score(y_test,y_pred))

classifier_entropy1=DecisionTreeClassifier(criterion='entropy',min_samples_split=50)
classifier_entropy1.fit(x_train,y_train)
y_pred_entropy1=classifier_entropy1.predict(x_test)
print('accuracy on test data using entropy:',accuracy_score(y_true=y_test,y_pred=y_pred_entropy1))

from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
dot_data=StringIO()
export_graphviz(classifier,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=data.feature_names,class_names=data.target_names)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
