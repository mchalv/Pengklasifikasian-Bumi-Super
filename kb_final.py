import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import xlsxwriter

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

feature_cols = ["Massa Planet", "Radius Planet"]

def importdata():
    balance_data = pd.read_excel("Data_Planet.xlsx")
    print ("Dataset Length: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)
    print ("Dataset: \n", balance_data)
    return balance_data

def splitdataset(balance_data):
    X = balance_data[feature_cols]
    y = balance_data['Tipe Planet']

    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 100)
    
    return X, Y, X_train, X_test, y_train, y_test

def train_using_gini(X_train, X_test, y_train):
    clf_gini = DecisionTreeClassifier(
            criterion = "gini",
            random_state = 100,
            max_depth=3,
            min_samples_leaf=5)
    
    clf_gini.fit(X_train, y_train)
    return clf_gini

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy : ", accuracy_score(y_test,y_pred) * 100)
    print("\nReport : \n", classification_report(y_test, y_pred))

data = importdata()
print(data.isnull().sum())
X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
clf_gini = train_using_gini(X_train, X_test, y_train)

print("\n==== Results: ====\n")

y_pred_gini = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred_gini)

print("\n==== Visualisasi Model Tree ====\n")

dot_data = StringIO()
export_graphviz(clf_gini, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names=feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('model.png')
Image(graph.create_png())

print("==== input ====\n")

wb = xlsxwriter.Workbook("Prediksi.xlsx")
sheet1 = wb.add_worksheet("Sheet1")

input_massa = float(input("Massa : "))
input_radius = float(input("Radius : "))

for i in range(4):
    if i == 0: sheet1.write(0, i, "Nama Planet")
    elif i == 1: sheet1.write(0, i, "Massa Planet")
    elif i == 2: sheet1.write(0, i, "Radius Planet")
    elif i == 3: sheet1.write(0, i, "Tipe Planet")

for i in range(4):
    if i == 0: sheet1.write(1, i, "X")
    elif i == 1: sheet1.write(1, i, input_massa)
    elif i == 2: sheet1.write(1, i, input_radius)
    elif i == 3: sheet1.write(1, i, "")

wb.close()

open_file = pd.read_excel("Prediksi.xlsx")
X_1 = open_file[feature_cols]
pred = prediction(X_1, clf_gini)
