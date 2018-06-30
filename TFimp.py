from pcg.imports2 import *
import re


#Data cleaning and preprocessing#
#-------------------------------#

#Load df from file
dframe = pd.read_csv("train.csv")
dframe.drop("PassengerId", inplace =True, axis = 1)
#Deak with NA values
dframe.isnull().sum()

dframe['Age'].fillna(dframe['Age'].median(), inplace=True)
dframe['Embarked'].fillna(dframe['Embarked'].mode()[0], inplace = True)
dframe['Fare'].fillna(dframe['Fare'].median(), inplace = True)


#Split Cabin column into floor and rooms columns. i.e "C130 C230" ---> C - 130 - 120
newframe = dframe.Cabin.str.split(" ", expand=True)
dframe.Cabin= dframe.Cabin.str.extract('(\w)')
newframe = newframe.apply(lambda x: x.str.extract('(\d+)'))
dframe = pd.concat([dframe, newframe],axis=1)

dframe.rename(columns={0: 'cabin#1',1: 'cabin#2',2: 'cabin#3',3: 'cabin#4'}, inplace = True)

#Groupby Ticket and then give each person the count of the group
grp = dframe.groupby("Ticket").size()  #could have also used dframe.Ticket.value_counts()
dframe['SameTicket']= dframe.Ticket.map(grp)

#Give group numbers for each group of same ticket
dframe["TicketGrp"]= pd.Categorical(dframe.Ticket).codes

#Make a column of Parch + SibSp = Family
dframe['FamilySize'] = dframe['Parch'] + dframe['SibSp']+1

#Make a column is alone
dframe['IsAlone'] = dframe['FamilySize'].dropna().apply(lambda x: 1 if x == 1 else 0)

#Make a Title column
dframe["Title"]= dframe.Name.str.extract('^.*?,\s(.*?)\.')
#CLean Title
validTitles = (dframe["Title"].value_counts()/dframe["Title"].shape[0])<0.1
dframe["Title"] = dframe["Title"].apply(lambda x: "Misc" if validTitles.loc[x] == True else x )

#Make a  FamilyName column
dframe["FamilyName"]= dframe.Name.str.extract('^(\w+)')


#Encoding
label = LabelEncoder()

data1 = dframe.copy(deep=True)
data1 = data1[data1["cabin#1"].notnull()]
data1.dropna(inplace =True, thresh  =175, axis =1 )
labels = data1.select_dtypes(include=['object']).columns.get_values()

for i in labels:
    data1[str(i)] = label.fit_transform(data1[i])


array = data1.values
X = array[:,1:]
Y = array[:,1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


train_labels = keras.utils.to_categorical(Y_train, 10)
test_labels = keras.utils.to_categorical(Y_validation, 10)



#Turn NA values to -1 in order to be used in learning algorithms
#dframe.fillna(-1, inplace = True)
