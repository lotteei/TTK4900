from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import seaborn as sns
import pandas as pd
import numpy as np


file_path = 'data/'


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
                                   bins_list_yv, n_list_zv, bins_list_zv, bins_list_azv,
                                   n_list_a, bins_list_a)]
        X_data.append(data_point)
        y_data.append((row['h2s']))


    X = np.array(X_data)
    y = np.array(y_data)

    nsamples, nx, ny = X.shape
    X = X.reshape(nsamples, nx*ny)

    return X, y


X_train, y_train = get_x_y_data( file_path + 'behaviour_data.csv')
X_test, y_test = get_x_y_data(file_path + 'test_behaviour_data.csv')


C = 100.0
h=.02

# Defining estimation models for regression
decision_tree_model = DecisionTreeRegressor(random_state=0)
random_forest_model = RandomForestRegressor(max_depth=2, random_state=0)
SVR_model = SVR() # RBF kernel

svrs = [ SVR_model, random_forest_model, decision_tree_model]
kernel_label = ["SVR with RBF kernel", "Random forest regressor", "Decision tree regressor"]

for i, svr in enumerate(svrs):
    # Train models
    svr.fit(X_train, y_train)
    predictions = svr.predict(X_test)
    
    print('\n-----------------------------------------------------')
    print('Model - ', kernel_label[i])
    print("R2 Score          : ", r2_score(y_test, predictions))
    print("Mean_abs_error    : ", mean_absolute_error(y_test, predictions))
    print("Mean_sqr_error    : ", mean_squared_error(y_test, predictions, squared=False))

    # Plot regression plot
    fig = plt.figure(figsize=(9,9))
    ax= plt.subplot()
    sns.regplot(x=y_test, y=predictions)
    plt.xlabel('True $H_2S$ ($\mu g/L$)', fontsize=18)
    plt.ylabel('Predicted $H_2S$ ($\mu g/L$)', fontsize=18) 
    plt.title(kernel_label[i] + '\n$(R²='+str(round(r2_score(y_test, predictions),3))+')$', fontweight='bold', fontsize=22) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()
    
