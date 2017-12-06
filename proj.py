
from pcg.imports import *
from pcg.funcs import *

dframe = pd.read_csv("train.csv")
#Binning
bins= np.linspace(dframe["Age"].min()-1,dframe["Age"].max()+1,11)
group_names = np.linspace(1,10,10)
categories = pd.cut(dframe['Age'], bins, labels=group_names)
dframe['Age_grp'] = (categories)
dframe['Age_grp']= dframe['Age_grp'].astype("int64")


sex_groups = dframe.groupby("Sex") #groupby sex
sex_groups.groups #display groups
male_frame = sex_groups.get_group('male') #get a specific group
female_frame = sex_groups.get_group('female')

female_frame = female_frame.groupby("Age_grp", as_index=False).count().reset_index(level=0)[["Survived","Age_grp"]]
male_frame =male_frame.groupby("Age_grp",as_index=False).count().reset_index(level=0)[["Survived","Age_grp"]]


female_frame["Sex"] = "Female"
male_frame["Sex"] = "Male"


merged =female_frame.merge(male_frame,on = "Sex", how = "outer")
pivoted = merged.pivot("Sex","Age_grp_", "Survived" )

pivoted['index1'] = pivoted.index
a

dframe['index1'].dtypes





