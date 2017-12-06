
from pcg.imports import *
from pcg.funcs import *

dframe = pd.read_csv("train.csv")

#Binning
bins= np.linspace(dframe["Age"].min()-1,dframe["Age"].max()+1,11)
group_names = np.linspace(1,10,10)
categories = pd.cut(dframe['Age'], bins, labels=group_names)
dframe['Age_grp'] = (categories)
dframe['Age_grp']= dframe['Age_grp']


sex_groups = dframe.groupby("Sex",as_index=False) #groupby sex
sex_groups.groups #display groups
male_frame = sex_groups.get_group('male') #get a specific group
female_frame = sex_groups.get_group('female')






