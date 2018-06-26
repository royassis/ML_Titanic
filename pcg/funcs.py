import numpy as np
from pandas import DataFrame
import re

def ranker (df):
    df["A"]  = np.arange(len(df)) + 1
    return df

def extracter (row):
    return  re.match('\d+|$', str(row))[0]
