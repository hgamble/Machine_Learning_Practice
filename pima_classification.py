# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:50:09 2017

@author: HGAMBLE
"""

from sklearn import tree as t
import pandas as pd
import numpy as np

def train_test_tree_fit(df):
    df_shuffle = df.sample(frac=1).reset_index(drop = True)[:int(df.size*.7)]
    df_t = df_shuffle[:int(len(df.index)*.7)]
    df_t_data = df_t.drop(df_t.columns != "response")
    df_t_response = df_t.drop(df_t.columns != "response")
    df_v = df_shuffle[int(len(df.index)*.7):]
    clf = t.DecisionTreeClassifier()
    clf.fit(df_t_data, df_t_response)
    


def main():
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data", 
                     header = None, index_col = False)
    df.columns = ["n_pregnancies",  "pl_gluscose_conc", "blood_pressure",
                  "tri_fold_thick", "insulin_level", "bmi", "dpf", "age", "response"]
    df["response"] = df["response"].astype('category')
    df.hist(column ="n_pregnancies")
    for col in df.columns:
        print(df[col].dtype)
    #train_test_tree_fit(df)
    #df.dropna(inplace = True)
    #print(df)


if __name__ == '__main__':
    main()