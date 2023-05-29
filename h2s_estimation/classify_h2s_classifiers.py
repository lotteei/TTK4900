import pandas as pd
import numpy as np
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import time
import statistics
import json
import random
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn import preprocessing, utils

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


file_path = 'data/'

lab = preprocessing.LabelEncoder()

def get_x_y_data(file_path):
        
    df = pd.read_csv(file_path)
    X_data = []
    y_data = []
    for _, row in df.iterrows():

        res = row['n_v'].strip('][ ').split()
        n_list = [float(s) for s in res]
        res = row['bins_v'].strip('][ ').split()
        bins_list = [float(s) for s in res]

        res = row['n_xv'].strip('][ ').split()
        n_list_xv = [float(s) for s in res]
        res = row['bins_xv'].strip('][ ').split()
        bins_list_xv = [float(s) for s in res]

        res = row['n_yv'].strip('][ ').split()
        n_list_yv = [float(s) for s in res]
        res = row['bins_yv'].strip('][ ').split()
        bins_list_yv = [float(s) for s in res]

        res = row['n_zv'].strip('][ ').split()
        n_list_zv = [float(s) for s in res]
        res = row['bins_zv'].strip('][ ').split()
        bins_list_zv = [float(s) for s in res]

        
        bins_list_azv = [float(s) for s in res]
        res = row['n_a_v'].strip('][ ').split()
        n_list_a = [float(s) for s in res]
        res = row['bins_a_v'].strip('][ ').split()
        bins_list_a = [float(s) for s in res]
    

        data_point=[a for a in zip(n_list, bins_list, n_list_xv, bins_list_xv, n_list_yv,
                                   bins_list_yv, n_list_zv, bins_list_zv,bins_list_azv,
                                   n_list_a, bins_list_a)]
        X_data.append(data_point)
        y_data.append(float(row['h2s']))


    X = np.array(X_data)
    y = np.array(y_data)
    y_trainsformed = lab.fit_transform(y)

    nsamples, nx, ny = X.shape
    X = X.reshape(nsamples, nx*ny)
    return X, y_trainsformed


X_train, y_train = get_x_y_data(file_path + 'behaviour_data.csv')
X_test, y_test = get_x_y_data(file_path + 'test_behaviour_data.csv')
# Create the SVM models
C = 1.0
h=.02



# Define classification models
lin_svc = svm.SVC(kernel='linear')
rbf_svc = svm.SVC(kernel='rbf',  C=C)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier(random_state=0)



# title for the plots
titles = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial kernel $(d = 3)$', 
          'Decision tree classifier', 'Random forest classifier']

svrs = [lin_svc, rbf_svc,  poly_svc, decision_tree_model, random_forest_model]

# Training models
for i, svr in enumerate(svrs):
    svr.fit(X_train, y_train)
    predict = svr.predict(X_test)
    predict = lab.inverse_transform(predict)

    print('\n---------------------------------------------')
    print('Model - ', titles[i])
    print('True: ', [i for i in lab.inverse_transform(y_test)])
    print('Pred: ', [(i) for i in predict])



# Predict with models
for i, clf in enumerate(svrs):
    y_pred=[]
    predictions = clf.predict(X_test)

    print('\n--------------------------------------\n', titles[i])
    print("Accuracy score: ", accuracy_score(y_test, predictions), "\n")
 
    
    pred_set=set(lab.inverse_transform(y_test))
    new_labels = sorted(pred_set)
    labels=['False', 'True']
    
    # Plot confusion matrix
    fig = plt.figure(figsize=(15,10))
    ax= plt.subplot()
    cm = confusion_matrix(y_test,predictions)
    sns.set(font_scale=1.5) 
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.1%', cmap='Blues')
    ax.set_xlabel('Predicted labels', fontsize=18);ax.set_ylabel('True labels', fontsize=18) 
    ax.set_title('Confusion matrix for '+ titles[i], fontweight='bold', fontsize=22) 
    ax.xaxis.set_ticklabels(new_labels, fontsize=12); ax.yaxis.set_ticklabels(new_labels, fontsize=12)
    plt.show()
    
