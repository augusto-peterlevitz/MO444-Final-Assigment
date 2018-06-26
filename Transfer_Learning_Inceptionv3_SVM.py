import os
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
import os
kernel = 'rbf'
train_path = r'\Users\Documents\assigment_05\transfer_learning'
val_path = r'\Users\Documents\assigment_05\transfer_learning'
test_path = r'\Users\Documents\assigment_05\transfer_learning'

X_train_path = os.path.join(train_path, 'X_train.csv')
y_train_path = os.path.join(train_path, 'y_train.csv')
X_train = np.genfromtxt(X_train_path,delimiter=',')
y_train = np.genfromtxt(y_train_path,delimiter=',', dtype='str')

X_val_path = os.path.join(val_path, 'X_val.csv')
y_val_path = os.path.join(val_path, 'y_val.csv')
X_val = np.genfromtxt(X_val_path,delimiter=',')
y_val = np.genfromtxt(y_val_path,delimiter=',',dtype='str') 


X_test_path = os.path.join(test_path, 'X_test.csv')
y_test_path = os.path.join(test_path, 'y_test.csv')
X_test = np.genfromtxt(X_test_path,delimiter=',')
y_test = np.genfromtxt(y_test_path,delimiter=',',dtype='str') 

svm = SVC(kernel = kernel, C = 1, gamma = 'auto', decision_function_shape = 'ovo', probability=True)

fitting=svm.fit(X_train, y_train)
acc = svm.score(X_val, y_val)
print(acc)
acc_test = svm.score(X_test, y_test)
print(acc_test)


#to save:
from sklearn.externals import joblib
joblib.dump(svm, r'\Users\Documents\assigment_05\transfer_learning\SVM_DATA_FITTED.pkl')
