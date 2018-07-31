from pcg.imports_n_settings import *
from pcg.functions import *
from pcg.NNmodel import *
from keras.optimizers import SGD, RMSprop

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

"""
#Cross validation
def create_model(optimizer='adam',lr=0.0001, loss= 'binary_crossentropy',
                 layer_a=50, layer_b=50,layer_c=50,
                 activation_a = 'relu' ,activation_b ='relu',activation_c ='relu'):

    # create model
    model = Sequential()
    model.add(Dense(layer_a, input_dim=dummy1.shape[1], activation=activation_a))
    model.add(Dense(layer_b, activation=activation_b))
    model.add(Dense(layer_c, activation=activation_c))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    #optimzer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model
"""
"""
model_w_Wrapper = KerasClassifier(build_fn=create_model2, epochs=1000, batch_size = dummy1.shape[0],verbose=0)

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
"""


# Hyperparameters search

# define the grid search parameters
model_w_Wrapper = KerasClassifier(build_fn=create_model2, epochs=300, batch_size = 500,verbose=1)

#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#optimizer = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
optimizer = [SGD, RMSprop,Adam, Adamax]
#loss = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",\
#        "mean_squared_logarithmic_error", "squared_hinge","hinge","categorical_hinge",\
#        "logcosh","categorical_crossentropy", "sparse_categorical_crossentropy",\
#        "binary_crossentropy", "kullback_leibler_divergence", "poisson", "cosine_proximity"]
loss = ["mean_squared_error","binary_crossentropy"]

layer_a= layer_b = layer_c = np.arange(10,40, step=10)
activation_a = activation_b = activation_c = ["relu", "sigmoid"]
lr = np.linspace(0.000001,0.0001,4)
momentum = np.linspace(0.01 ,0.0001,4)

#param_grid = dict(optimizer=optimizer, loss= loss, layer_a =layer_a, layer_b =layer_b, layer_c=layer_c,
#                  activation_a=activation_a, activation_b=activation_b, activation_c=activation_c,
#                  lr=lr,momentum=momentum)

param_grid = dict(optimizer=optimizer, loss= loss)


grid = GridSearchCV(estimator=model_w_Wrapper, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)

grid_result.cv_results_['mean_train_score'][grid_result.best_index_]
grid_result.cv_results_['params'][grid_result.best_index_]



model_w_Wrapper = KerasClassifier(build_fn=create_model2, epochs=1000, batch_size = 10,verbose=1)
model_w_Wrapper.fit(X_train, y_train)
a = cross_val_score(model_w_Wrapper,X_train, y_train)


#fit model and predict
model = create_model2()
model.fit(X,Y,epochs=1000,batch_size = 1, verbose=0)

prediction = model_w_Wrapper.predict(dummy2).argmax(axis = 1)

data2["Survived"]= prediction




#Save to file
submission = data2[["PassengerId","Survived"]]
dir ="predictions"

savetofile(submission, cross_val_score_results, dir)

logfile(create_model,cross_val_score_results, dummy1.columns.values)