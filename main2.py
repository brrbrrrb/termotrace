import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# from matplotlib.figure import Figure
from pandas import ExcelWriter

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier


import sys
import os
import io
from datetime import datetime
import streamlit as st
import base64


print(os. getcwd())

def predict_y(train_file_name, test_file_name):
    print(train_file_name, test_file_name)

    # train_data = pd.read_excel('my_data\\data_0.1.xlsx')
    train_data = pd.read_excel(train_file_name)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1:].values
    time_arr_train = X_train[:, 0]

    zones_train_vals = list(y_train.ravel())
    zones_train = sorted(list(set(zones_train_vals)))

    # test_data = pd.read_excel('my_data\\data_0.3_scale_3.21.xlsx')
    test_data = pd.read_excel(test_file_name)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1:].values

    zones_test_vals = list(test_data.iloc[:,-1].values)
    zones_test = sorted(list(set(zones_test_vals)))
    
    # fig = plt.figure(dpi=200)
    # axes = fig.add_axes([0,0,1,1])
    # axes.scatter(X[:, 0], X[:, 1], color='red', marker=',', lw=0, s=1)
    # axes.scatter(X2[:, 0], X2[:, 1], color='green', marker='.')
    
    # fig, ax = plt.subplots()
    # plt.scatter(X_test[:, 0], X_test[:, 1], color='green', marker='.', label="test data")
    # plt.legend()
    # plt.xlabel("time")
    # plt.ylabel("temperature")
    # plt.tight_layout()
    # for i, category in enumerate(zones_train[1:]):
    #     start_index = zones_train_vals.index(category)
    #     plt.axvline(time_arr_train[start_index], color='black', linestyle=':')
    # buffer_before_sc_pred = io.BytesIO()
    # plt.savefig(buffer_before_sc_pred, format='png')
    # plt.clf()
    
    plt.scatter(X_train[:, 0], X_train[:, 1], color='red', marker='.', label="train data")
    plt.scatter(X_test[:, 0], X_test[:, 1], color='green', marker='.', label="test data")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("temperature")
    plt.tight_layout()
    buffer_before_sc = io.BytesIO()
    plt.savefig(buffer_before_sc, format='png')
    plt.clf()

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    plt.scatter(X_train[:, 0], X_train[:, 1], color='red', marker='.', label="train data")
    plt.scatter(X_test[:, 0], X_test[:, 1], color='green', marker='.', label="test data")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("temperature")
    plt.tight_layout()
    buffer_after_sc = io.BytesIO()
    plt.savefig(buffer_after_sc, format='png')
    plt.clf()

    names = [
        "LogisticRegression",
        "KNeighbors",
        "SVC",
        "GaussianNB",
        "DecisionTree",
        "RandomForest",
        "GaussianProcess",
        "MLP",
        "AdaBoost",
        "QDA", #QuadraticDiscriminantAnalysis
        "XGB",
    ]

    classifiers = [
        LogisticRegression(),
        KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
        SVC(kernel = 'sigmoid', random_state = 0),
        GaussianNB(),
        DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
        RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
        GaussianProcessClassifier(random_state = 0),
        MLPClassifier(alpha=1, max_iter=1000, random_state=0),
        AdaBoostClassifier(random_state=0),
        QuadraticDiscriminantAnalysis(),
        XGBClassifier(),
    ]

    acc_score_arr = []
    predict_vals_arr = [y_test.reshape(1, -1)[0]]


    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train.ravel())
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        acc_score = accuracy_score(y_test, y_test_pred)
        acc_score_arr.append([name, acc_score])
        predict_vals_arr.append(y_test_pred)
        
    acc_score_arr.append(['', ''])
    acc_score_arr.append(['Train excel', train_file_name.split('/')[-1]])
    acc_score_arr.append(['Test excel', test_file_name.split('/')[-1]])
    acc_score_df = pd.DataFrame(data=acc_score_arr, columns = ['Model Name','Accuracy Score'])


    predict_vals_arr = np.array(predict_vals_arr)
    predict_vals_arr = predict_vals_arr.transpose()
    # print(predict_vals_arr.shape)

    predict_vals_df = pd.DataFrame(data=predict_vals_arr, columns = ['True vals'] + names)


    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # print("Current date & time : ", current_datetime)
    
    str_current_datetime = str(current_datetime)
    
    file_name = f"excel_predict\\predict_{str_current_datetime}.xlsx"
    

    with pd.ExcelWriter(file_name) as excel_writer:
        acc_score_df.to_excel(excel_writer, sheet_name='accuracy_score', index=False)
        predict_vals_df.to_excel(excel_writer, sheet_name='predict_vals', index=False)
        

    return (file_name, buffer_before_sc, buffer_after_sc, acc_score_df)


if __name__ == '__main__':

    st.write('''
             1. Виберіть файл із тренувальними даними
             2. Виберіть файл із тестувальними даними
             3. Натисніть **Submit** ''')
    uploaded_file1 = st.file_uploader("Choose train excel:")
    uploaded_file2 = st.file_uploader("Choose test excel:")
    bt = st.button('Submit')
    
    if (uploaded_file1 is not None) and (uploaded_file2 is not None) and bt:
        # To read file as bytes:
        # st.write(uploaded_file1.name)
        # st.write(uploaded_file2.name)
        # st.write(os. getcwd())
        
        predict_file_name_full, fig1_bytes, fig2_bytes, acc_score_df = predict_y("excel_data\\"+uploaded_file1.name, "excel_data\\"+uploaded_file2.name)
        
        predict_file_name = predict_file_name_full.split('\\')[-1]
        st.header('Data before standard scaler:')
        st.image(fig1_bytes)
        
        st.header('Data after standard scaler:')
        st.image(fig2_bytes)
        
        st.header('Results score:')
        st.dataframe(acc_score_df, height=550)
        
        st.header('Download excel with results:')
        with open(predict_file_name_full, 'rb') as file:
            st.download_button(
                label=predict_file_name,
                data=file,
                file_name=predict_file_name,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        

