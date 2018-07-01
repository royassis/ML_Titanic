from pcg.imports2 import *
from collections import defaultdict


#Data cleaning and preprocessing#
#-------------------------------#
#Load df from file
dframe = pd.read_csv("train.csv")

#Deal with NA values
dframe.isnull().sum()

dframe['Age'].fillna(dframe['Age'].median(), inplace=True)
dframe['Embarked'].fillna(dframe['Embarked'].mode()[0], inplace = True)
dframe['Fare'].fillna(dframe['Fare'].median(), inplace = True)
dframe.drop("Cabin", inplace =True, axis = 1)

#Make a column of Parch + SibSp = Family
dframe['FamilySize'] = dframe['Parch'] + dframe['SibSp']+1

#Make a column is alone
dframe['IsAlone'] = dframe['FamilySize'].dropna().apply(lambda x: 1 if x == 1 else 0)

#Make a Title column
dframe["Title"]= dframe.Name.str.extract('^.*?,\s(.*?)\.')
#CLean Title
validTitles = (dframe["Title"].value_counts()/dframe["Title"].shape[0])<0.1
dframe["Title"] = dframe["Title"].apply(lambda x: "Misc" if validTitles.loc[x] == True else x )

#Binning relevent columns
dframe['AgeBin'] = pd.cut(dframe['Age'].astype(int), 5)
dframe['FareBin'] = pd.qcut(dframe['Fare'], 4)

############Encoding#############
#-------------------------------#

#Get relevent columns to encode
columnsToEnc=['Sex', 'Title', 'AgeBin', 'FareBin', "Embarked"]

#Create a dictionary containing encoders
d = defaultdict(LabelEncoder)
dframe[columnsToEnc].apply(lambda x: d[x.name].fit(x))
for i in columnsToEnc:
    dframe[i.__str__()+"_Code"]=columnsToEnc.transform(dframe[i])



Target  = ['Survived']
dframe_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
dframe_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']
dframe_xy = Target + dframe_x


#Split Training and Testing Data#
#-------------------------------#


train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(dframe[dframe_x_calc] ,dframe[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(dframe[dframe_x_bin], dframe[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(dframe_dummy[dframe_x_dummy], dframe[Target], random_state = 0)














array = dframe.values
X = np.concatenate ((array[:,1:10], array[:,11:]), axis =1)
Y = array[:,10]
validation_size = 0.10
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


train_labels = keras.utils.to_categorical(Y_train, 7)
test_labels = keras.utils.to_categorical(Y_validation, 7)


model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(17,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax'))



model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


history = model.fit(X_train, train_labels,
                    batch_size=156,
                    epochs=1000,
                    verbose=2,
                    validation_data=(X_validation, test_labels)
                    )


score = model.evaluate(X_validation, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])






#Turn NA values to -1 in order to be used in learning algorithms
#dframe.fillna(-1, inplace = True)


# Encode whole dframe without NA values but keep dframe whole
# Pass TF only dframe without NA values






"""
#Split Cabin column into floor and rooms columns. i.e "C130 C230" ---> C - 130 - 120
newframe = dframe.Cabin.str.split(" ", expand=True)
dframe.Cabin= dframe.Cabin.str.extract('(\w)')
newframe = newframe.apply(lambda x: x.str.extract('(\d+)'))
dframe = pd.concat([dframe, newframe],axis=1)

dframe.rename(columns={0: 'cabin#1',1: 'cabin#2',2: 'cabin#3',3: 'cabin#4'}, inplace = True)

#Groupby Ticket and then give each person the count of the group
grp = dframe.groupby("Ticket").size()  #could have also used dframe.Ticket.value_counts()
dframe['SameTicket']= dframe.Ticket.map(grp)

#Make a  FamilyName column
dframe["FamilyName"]= dframe.Name.str.extract('^(\w+)')

#Give group numbers for each group of same ticket
dframe["TicketGrp"]= pd.Categorical(dframe.Ticket).codes

dframe[labels] = dframe[labels].apply(lambda x: d[x.name].fit_transform(x))
"""