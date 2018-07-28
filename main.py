from pcg.imports_n_settings import *
from pcg.functions import *

#Data cleaning and preprocessing#
#-------------------------------#
#Load df from file
data1 = pd.read_csv("train.csv")
data2 = pd.read_csv("test.csv")

sets = [data1,data2]

#Deal with NA values
data1.isnull().sum()

for set in sets:
    set['Age'].fillna(set['Age'].median(), inplace=True)
    set['Embarked'].fillna(set['Embarked'].mode()[0], inplace = True)
    set['Fare'].fillna(set['Fare'].median(), inplace = True)
    set.drop("Cabin", inplace =True, axis = 1)

#Make a Title column
for set in sets:
    set["Title"]= set.Name.str.extract('^.*?,\s(.*?)\.', expand = False)
    #CLean Title
    validTitles = (set["Title"].value_counts()/set["Title"].shape[0])<0.1
    set["Title"] = set["Title"].apply(lambda x: "Misc" if validTitles.loc[x] == True else x )

#Binning relevent columns
for set in sets:
    set['AgeBin'] = pd.cut(set['Age'].astype(int), 5) #  pd.cut(set['Age'].astype(int), 4, labels = (0,1,2,3), retbins=True)[1]  ;;    data1['AgeBin'].unique()
    set['FareBin'] = pd.qcut(set['Fare'], 4) # pd.qcut(set['Fare'], 4, labels =(0,1,2,3), retbins = True)[1]

    set['FareBin'].dtypes.categories.values.astype(str)

#Y column
Target = ['Survived']

#One hot data
dummy_sets = []
for set in sets:
    dummy = pd.get_dummies(set[['AgeBin', 'FareBin', 'Sex', 'Embarked', 'Title',"Pclass","SibSp","Parch"]])
    dummy = dummy.astype(int)
    dummy_sets.append(dummy)

dummy1 = dummy_sets[0]
dummy2 = dummy_sets[1]

#Split to train and test
X_train, X_test, y_train, y_test =train_test_split(dummy1.values, data1[Target])
#Dummy y values
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
X= dummy1.values
Y= keras.utils.to_categorical(data1[Target])

#Cross validation
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=dummy1.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    optimzer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=['accuracy'])
    return model

model_w_Wrapper = KerasClassifier(build_fn=create_model, epochs=1000, batch_size = dummy1.shape[0],verbose=0)

#Model selection
models = {"NN":model_w_Wrapper,
          "Tree":DecisionTreeClassifier()
          }

cross_val_score_results={}
for model in models:
    results = cross_val_score(models[model], X_test, y_test)
    #cross val results
    accu = np.mean(results)
    cross_val_score_results[model]=accu

print(cross_val_score_results)


#fit model and predict
model = create_model()
model.fit(X,Y,epochs=1000,batch_size = dummy1.shape[0], verbose=0)

prediction = model.predict(dummy2).argmax(axis = 1)

data2["Survived"]= prediction


#Save to file
submission = data2[["PassengerId","Survived"]]
dir ="predictions"

savetofile(submission, cross_val_score_results, dir)

