import os
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

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

RF_srqt_oob_error = []
RF_log2_oob_error = []
RF_None_oob_error = []
number_of_trees = []
for n_trees in range(10, 251):
    RF_sqrt = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=n_trees, n_jobs=-1, oob_score=True, random_state=None,
                verbose=1, warm_start=False)
    fitting = RF_sqrt.fit(X_train, y_train)
    oob_error = 1 - RF_sqrt.oob_score_
    RF_srqt_oob_error.append(oob_error)

np.savetxt('RF_srqt_oob_error.csv', RF_srqt_oob_error, delimiter=',')

for n_trees in range(10, 251):    
    RF_log2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='log2', max_leaf_nodes=None,
                min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=n_trees, n_jobs=-1, oob_score=True, random_state=None,
                verbose=1, warm_start=False)
    fitting = RF_log2.fit(X_train, y_train)
    oob_error = 1 - RF_log2.oob_score_
    RF_log2_oob_error.append(oob_error)

np.savetxt('RF_log2_oob_error.csv', RF_log2_oob_error, delimiter=',')

for n_trees in range(10, 251):    
    RF_None = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features=None, max_leaf_nodes=None,
                min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=n_trees, n_jobs=-1, oob_score=True, random_state=None,
                verbose=1, warm_start=False)
    fitting = RF_None.fit(X_train, y_train)
    oob_error = 1 - RF_None.oob_score_
    RF_None_oob_error.append(oob_error)

np.savetxt('RF_None_oob_error.csv', RF_None_oob_error, delimiter=',')
RF_log2_oob_error = np.genfromtxt('RF_log2_oob_error.csv',delimiter=',')
RF_srqt_oob_error = np.genfromtxt('RF_srqt_oob_error.csv',delimiter=',')
number_of_trees = []
for n_trees in range(10, 251):
    number_of_trees.append(n_trees)
    
plt.plot(number_of_trees, RF_srqt_oob_error, 'b', label='max_features = sqrt(n_features)')
plt.plot(number_of_trees, RF_log2_oob_error, 'r', label='max_features = log2(n_features)')
plt.title('Random Forests: Evaluation of the Number of Trees')
plt.ylabel('Out-of-Bag Error Rate')
plt.xlabel('Number Of Trees')
plt.legend()
plt.figure()
plt.show()

RF_sqrt = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=250, n_jobs=-1, oob_score=True, random_state=None,
                verbose=1, warm_start=False)
fitting = RF_sqrt.fit(X_train, y_train)
RF = RF_sqrt
acc = RF.score(X_val, y_val)
print(acc)
acc_test = RF.score(X_test, y_test)
print(acc_test)

#to save:
from sklearn.externals import joblib
joblib.dump(RF, r'\Users\Documents\assigment_05\transfer_learning\RandomForests_DATA_FITTED.pkl')

