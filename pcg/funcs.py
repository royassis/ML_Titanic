
import re
import numpy as np
import random

#Functions to be used in pandas apply#
#------------------------------------#

def ranker (df):
    df["A"]  = np.arange(len(df)) + 1
    return df

def extracter (row):
    return  re.match('\d+|$', str(row))[0]


def normalized(arr):
    arr.astype(float)
    arrC= arr.copy()
    for x in range(len(arr)):
        arrC[x] = arr[x]/arr.sum()
    return arrC


def votePrec (arr):

    arr = np.array(arr)

    while(True):
        normalized(arr)
        cutoff = arr.max()
        lott = random.uniform(0, 1)

        if cutoff > lott:
            return arr.argmax()
        else:
            arr[arr.argmax()] = 0
            normalized(arr)


import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



