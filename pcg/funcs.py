import numpy as np
import re


#Functions to be used in pandas apply#
#------------------------------------#

def ranker (df):
    df["A"]  = np.arange(len(df)) + 1
    return df

def extracter (row):
    return  re.match('\d+|$', str(row))[0]
