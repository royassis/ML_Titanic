
from pcg.imports import *
from pcg.funcs import *

#importing Dataset
dframe = pd.read_csv("train.csv")
dframe.drop_duplicates(inplace=True)
dframe.dtypes


#Binning
bins= np.linspace(dframe["Age"].min()-1,dframe["Age"].max()+1,11)
group_names = np.linspace(1,10,10)
categories = pd.cut(dframe['Age'], bins, labels=group_names)
dframe['Age_grp'] = (categories)

# extract titles
dframe["Title"]= dframe["Name"].astype("str").str.extract("([a-zA-Z]+)\.",expand=False)

###
#extract multiple cabin values to new columns

#Extract cabin values to new dataframe
d2 = dframe["Cabin"].astype("str").str.extractall('([a-zA-Z])(\d+)').unstack()

#change None to NA
d2.fillna(value=np.nan, inplace=True)

#set a new multiindex for d2
idx = pd.MultiIndex.from_tuples([(1, u'one'), (1, u'two'),(1, u'three'),(1, u'four'),
                                 (2, u'one'), (2, u'two'), (2, u'three'), (2, u'four')],
                                  names=[None,None])

# flatten multiindex
d2.columns = d2.columns.get_level_values(0)

#change column names
d2.columns =["Cabin_l1","Cabin_l2","Cabin_l3","Cabin_l4","Cabin_#1","Cabin_#2","Cabin_#3","Cabin_#4"]

#change order of columns
d2 = d2[["Cabin_l1","Cabin_#1","Cabin_l2","Cabin_#2","Cabin_l3","Cabin_#3","Cabin_l4","Cabin_#4"]]

#merge to old dframe
d3 = d2.merge(dframe, right_index  = True, left_index  = True, how = "outer")

# change order of columns
d3 = d3[['PassengerId', 'Survived', 'Pclass', 'Name',"Title", 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin','Cabin_l1', 'Cabin_#1', 'Cabin_l2', 'Cabin_#2', 'Cabin_l3', 'Cabin_#3', 'Cabin_l4', 'Cabin_#4',  'Embarked', 'Age_grp']]

# change to float
for x in range(4):
    d3["Cabin_#"+str(x+1)]= pd.to_numeric(d3["Cabin_#"+str(x+1)])



"""

for index, row in dframe.iterrows():
    dframe.iloc[index, 12]= (str(dframe.iloc[index, 10]).split(" ")) + dframe.iloc[index, 12]

re.match('\w', "c100").group()
re.match('\d+', "c100")


#example of how to get all values of a cabin 
dframe["Cabin"].astype("str").str.extractall('([a-zA-Z])(\d+)').loc[27]

#get all entries that survived and groupby sex, count and than get the first location 
dframe".loc[dframe["Survived"] ==1].groupby("Sex").count().iloc[0,1]


#present bar plot
sns.barplot(dframe["Sex"], dframe["Survived"] )
"""
