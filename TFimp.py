from pcg.imports2 import *
from collections import defaultdict
from pcg.funcs import votePrec
import re


#Data cleaning and preprocessing#
#-------------------------------#

#Load df from file
dframe = pd.read_csv("train.csv")

#Deal with NA values
dframe.isnull().sum()

dframe['Age'].fillna(dframe['Age'].median(), inplace=True)
dframe['Embarked'].fillna(dframe['Embarked'].mode()[0], inplace = True)
dframe['Fare'].fillna(dframe['Fare'].median(), inplace = True)

dataCopy = dframe.copy(deep=True)

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
dataCopy.drop("Cabin", inplace =True, axis = 1)
labels = dataCopy.select_dtypes(include=['object']).columns.get_values()

#Create a dictionary containing encoders
d = defaultdict(LabelEncoder)
dataEncoded = dataCopy[labels].apply(lambda x: d[x.name].fit_transform(x))


array = dataCopy.values
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