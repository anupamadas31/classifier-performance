from __future__ import print_function
import pandas as pd
import numpy as np
import random
import itertools
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
np.set_printoptions(threshold=np.inf)
import collections
from sklearn.metrics import accuracy_score

# #load dataset
# df =pd.read_csv(r'D:/path/to/data.csv')
#
# #choosing features through column numbers
# X2= df.iloc[:,[1,2,3,5,7,8,9,11] ].values
# Y2=df.iloc[:,13].values#target value
#
# classify=RandomForestClassifier(n_estimators=10, max_depth=8,min_samples_split=5,bootstrap=False)#the model
#
# #evaluating the model through cross validation
# seed = 7
# np.random.seed(seed)
# kfold = KFold(n_splits=10,random_state=seed)
# Y_Predicted = cross_val_predict(classify, X, Y, cv=kfold)
# # prediction = pd.DataFrame(Y_Predicted, columns=['predictions']).to_csv('prediction_rf.csv')
# classify.fit(X,Y)
# score_folds=cross_val_score(classify, X,Y,cv=10)
# print(score_folds)
# acc=(accuracy_score(Y, Y_Predicted) * 100)
# print(acc)


confusionMatrix = confusion_matrix(Y, Y_Predicted)
# print(confusionMatrix)


############plotting the confusionn matrix#############
def plot_confusion_matrix(cm, target_names, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title('Random Forest,kfold=10',size=15)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape
    fmt =  'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",size= 20,style='oblique')

    plt.ylabel('True label',size=15)
    plt.xlabel('Predicted label',size=15)


# plt.figure()
plot_confusion_matrix(confusionMatrix, target_names=['1','2','3'])
plt.show()


#################accuracy across folds########################
# # classify.fit(X2,Y2)
# score_folds=cross_val_score(classify, X2,Y2,cv=10)
# print(score_folds)
# acc=(accuracy_score(Y2, Y_Predicted) * 100)
# print(acc)






#
