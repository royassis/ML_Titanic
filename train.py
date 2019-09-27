from pcg.imports_and_settings import *
from pcg.functions import *
from collections import defaultdict


#Data cleaning and preprocessing#
#-------------------------------#
#Load df from file
data1 = pd.read_csv("train.csv")

#Deal with NA values
data1.isnull().sum()

data1['Age'].fillna(data1['Age'].median(), inplace=True)
data1['Embarked'].fillna(data1['Embarked'].mode()[0], inplace = True)
data1['Fare'].fillna(data1['Fare'].median(), inplace = True)
data1.drop("Cabin", inplace =True, axis = 1)

#Make a column of Parch + SibSp = Family
data1['FamilySize'] = data1['Parch'] + data1['SibSp']+1

#Make a column is alone
data1['IsAlone'] = data1['FamilySize'].dropna().apply(lambda x: 1 if x == 1 else 0)

#Make a Title column
data1["Title"]= data1.Name.str.extract('^.*?,\s(.*?)\.', expand = False)
#CLean Title
validTitles = (data1["Title"].value_counts()/data1["Title"].shape[0])<0.1
data1["Title"] = data1["Title"].apply(lambda x: "Misc" if validTitles.loc[x] == True else x )

#Binning relevent columns
data1['AgeBin'] = pd.cut(data1['Age'].astype(int), 5) #  pd.cut(data1['Age'].astype(int), 4, labels = (0,1,2,3), retbins=True)[1]  ;;    data1['AgeBin'].unique()
data1['FareBin'] = pd.qcut(data1['Fare'], 4) # pd.qcut(data1['Fare'], 4, labels =(0,1,2,3), retbins = True)[1]

data1['FareBin'].dtypes.categories.values.astype(str)

#How to change categories cat group
cat = data1['AgeBin'].cat.categories #save categories of one group
data1['FareBin'].cat.set_categories(cat) #apply if to another group
data1['AgeBin'].astype(str) #encode it and after the can decode it


x=60
array = pd.cut(data1['Age'].astype(int), 4, labels = (0, 1, 2, 3), retbins=True)[1]
def returnBin(arr,x):
    for i in range(len(array)-1):
        if array[i]<=x and x<=array[i+1] :
            return (i)
returnBin(array, x)


############Encoding#############
#-------------------------------#

#Get relevent columns to encode
columnsToEnc=['Sex', 'Title', 'AgeBin', 'FareBin', "Embarked"]


#Encode
label = LabelEncoder()


#Y column
Target = ['Survived']

#define x variables for original features aka feature selection
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')


#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')


#define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')










