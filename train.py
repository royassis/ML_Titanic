from pcg.imports_and_settings import *


#Load df from file
df = pd.read_csv(r"data\train.csv")

#Data cleaning and preprocessing#
#Deal with NA values#

#Display NA count per column
df.isnull().sum()

#Replace Age NA values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

#Replace Embarked NA values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

#Replace Fare NA values with median
df['Fare'].fillna(df['Fare'].median(), inplace = True)

#Drop Cabin column
df.drop("Cabin", inplace =True, axis = 1)


#Make a column of Parch + SibSp = Family
df['FamilySize'] = df['Parch'] + df['SibSp'] + 1


####Create new featuers####

#Make a column is alone
df['IsAlone'] = df['FamilySize'].dropna().apply(lambda x: 1 if x == 1 else 0)

#Make a Title column
df["Title"]= df.Name.str.extract('^.*?,\s(.*?)\.', expand = False)

#CLean Title
validTitles = (df["Title"].value_counts() / df["Title"].shape[0]) < 0.1
df["Title"] = df["Title"].apply(lambda x: "Misc" if validTitles.loc[x] == True else x)

#Binning relevent columns
df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)
df['FareBin'] = pd.qcut(df['Fare'], 4)
df['FareBin'].dtypes.categories.values.astype(str)


####Encode####
#Get relevent columns to encode
columnsToEnc=['Sex', 'Title', 'AgeBin', 'FareBin', "Embarked"]

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
data1_dummy = pd.get_dummies(df[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')










