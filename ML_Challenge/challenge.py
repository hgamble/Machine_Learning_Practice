# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 14:51:30 2018

@author: HGAMBLE
"""
import pandas as pd
import numpy as np
import math

def get_N(s):
    return sum(pd.notnull(s))
    
    
def get_pplus(s):
    answers = s[pd.notnull(s)]
    return float(sum(answers))/answers.size

def sum_score(s):
    return sum(s[pd.notnull(s)])

def get_student_scores(df_stud):
    df_stud['test_score'] = df_stud.apply(sum_score, axis = 1)
    return df_stud

def main():
    df = pd.DataFrame.from_csv("C://Users/hgamble/Desktop/ML_Challenge/astudentData.csv", index_col = None)
    df_raw = df.pivot_table(index='question_id', columns='user_id', values='correct')
    df_stud = df.pivot_table(index = "user_id", columns = "question_id", values = 'correct')
    df_stats = df_raw
    #df_stats['pplus'] = df_raw.apply(get_pplus, axis = 1)
    #df_stats['valid_n'] = df_raw.apply(get_N, axis = 1)
    student_scores = get_student_scores(df_stud)
    df_stats['point_biserial_correlation'] = student_scores.corr()['test_score'].drop('test_score')
    #print df_stats
    print df_stats.query('point_biserial_correlation > .20').shape
    

if __name__ == '__main__':
    main()