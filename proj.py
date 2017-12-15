
from pcg.imports import *
from pcg.funcs import *


dframe = pd.read_csv("train.csv")
dframe.dtypes


#Binning
bins= np.linspace(dframe["Age"].min()-1,dframe["Age"].max()+1,11)
group_names = np.linspace(1,10,10)
categories = pd.cut(dframe['Age'], bins, labels=group_names)
dframe['Age_grp'] = (categories)
dframe['Age_grp']= dframe['Age_grp']

sns.barplot(dframe["Sex"], dframe["Survived"] )



d2 = dframe["Cabin"].astype("str").str.extractall('([a-zA-Z])(\d+)')
dframe["Cabin"].astype("str").str.extractall('([a-zA-Z])(\d+)').loc[27]


dframe.loc[dframe["Survived"] ==1].groupby("Sex").count().iloc[0,1]



for index, row in dframe.iterrows():
    dframe.iloc[index, 12]= (str(dframe.iloc[index, 10]).split(" ")) + dframe.iloc[index, 12]

re.match('\w', "c100").group()
re.match('\d+', "c100")

p_testa = 0.1
p_testb = 0.5

ptesta_b = 0.1
