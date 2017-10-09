# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:50:09 2017

@author: HGAMBLE
"""

from sklearn import tree as t
from sklearn import metrics as m
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

def dataClean(df):
    df["response"] = df["response"].astype('category')
    df.drop(labels = ["tri_fold_thick", "insulin_level"], axis = 1, inplace = True)
    df = df[(df.pl_gluscose_conc != 0) & (df.blood_pressure != 0) & (df.bmi != 0) & (df.age != 0)]
    return df

def categorize(df):
    categories = {"n_pregnancies":[(0, 0, 2), (1, 2, 5), (2, 5, float('inf'))], 
                  "pl_gluscose_conc":[(0, 0.1, 95), (1, 95, 141), (2, 141, float('inf'))],
                  "blood_pressure":[(0, 0.1, 80), (1, 80, 90), (2, 90, float('inf'))],
                  "bmi":[(0, 0.1, 18.5), (1, 18.5, 25.1), (2, 25.1, 30.1), (3, 30.1, 35.1), (4, 35.1, float('inf'))],
                  "dpf":[(0, 0.1, 0.42), (1, 0.42, 0.82), (2, 0.82, float('inf'))],
                  "age":[(0, 0.1, 41), (1, 41, 60), (2, 60, float('inf'))]}
    for (idx, row) in df.iterrows():
        for col in row.index:
            if col != "response":
                levels = categories[col]
                for level in levels:
                    if float(row[col]) < level[2] and float(row[col]) >= level[1]:
                        #print(level[0])
                        df = df.set_value(idx, col, level[0])
        
    
def simple_fit(df):
    df_shuffle = df.sample(frac=1).reset_index(drop = True)[:int(df.size*.7)]
    df_t = df_shuffle[:int(len(df.index)*.7)]
    df_t_data = df_t.drop(labels = "response", axis = 1)
    df_t_response = df_t.drop(labels = [w for w in df_t.columns if w != "response"], axis = 1)
    df_v = df_shuffle[int(len(df.index)*.7):].drop(labels = "response", axis = 1)
    clf = t.DecisionTreeClassifier()
    clf.fit(df_t_data, df_t_response)
    pred = clf.predict(df_v)
    act = df_shuffle[int(len(df.index)*.7):].drop(labels = [w for w in df_t.columns if w != "response"], axis = 1)
    return m.accuracy_score(act, pred)
    
def cross_val(df):
    clf = t.DecisionTreeClassifier()
    X = df.drop(labels = "response", axis = 1)
    temp = np.array(df.drop(labels = [w for w in df.columns if w != "response"], axis = 1))
    y = temp.reshape(temp.shape[0])
    return cross_val_score(clf, X , y).mean()

def main():
    scores = {}
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data", 
                     header = None, index_col = False)
    df.columns = ["n_pregnancies",  "pl_gluscose_conc", "blood_pressure",
                  "tri_fold_thick", "insulin_level", "bmi", "dpf", "age", "response"]
    df = dataClean(df)
    categorize(df)
    scores["Simple Split Fit: "] = simple_fit(df)
    scores["Simple 5-Fold Cross Validation: "] = cross_val(df)
    print(scores)


if __name__ == '__main__':
    main()