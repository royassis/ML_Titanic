from os import walk
import re
import datetime
import numpy as np
import random
import pickle

#Define an iterative file name for scv files created
def savetofile(dframe, score_dict, dir):

    #Get a
    f = []
    for (dirpath, dirnames, filenames) in walk(dir):
        f.extend(filenames)
        break

    for filename in f:
        number = 0
        match = re.match("predictions_(\d+).*", filename)
        if match == None:
            continue
        if int(match.group(1)) > number:
            number = match.group(1)
    number = int(number) + 1

    filename = "predictions"
    number = number.__str__()
    date = datetime.date.today().strftime("%d-%m-%y")
    score = np.round(score_dict["NN"],2).__str__()
    suffix = "csv"

    str = '__'.join([filename, number, date, score])
    path = '/'.join([dir, str])
    path = '.'.join([path, suffix])

    dframe.to_csv(path, index=False)


def logfile(create_model,cross_val_score_results, fetures ):
    f = "predictions/log.txt"
    a = create_model()
    loss = str(a.loss)
    opt=str(a.optimizer.__class__)
    print_fn = lambda x: handle.write(x + '\n')

    with open(f, "a") as handle:
        a.summary(print_fn=print_fn)
        handle.write("loss= "+loss+",opt= "+opt+"\n")
        handle.write("opt= "+opt+"\n")
        handle.write("accuracy: " + str(cross_val_score_results)+"\n\n")
        handle.write("fetures "+str(fetures)+"\n")



#Functions to be used in pandas apply#
#------------------------------------#

def ranker (df):
    df["A"]  = np.arange(len(df)) + 1
    return df

def extracter (row):
    return  re.match('\d+|$', str(row))[0]

def normalized(arr):
    arr = arr.astype(float, copy=False)
    sum= arr.sum()
    for x in range(len(arr)):
        arr[x] = arr[x]/sum
    return arr


normalized(np.array([1,2,3]))

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

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
