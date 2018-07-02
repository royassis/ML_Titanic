from pcg.imports2 import *
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
data1["Title"]= data1.Name.str.extract('^.*?,\s(.*?)\.')
#CLean Title
validTitles = (data1["Title"].value_counts()/data1["Title"].shape[0])<0.1
data1["Title"] = data1["Title"].apply(lambda x: "Misc" if validTitles.loc[x] == True else x )

#Binning relevent columns
data1['AgeBin'] = pd.cut(data1['Age'].astype(int), 5)
data1['FareBin'] = pd.qcut(data1['Fare'], 4)

############Encoding#############
#-------------------------------#

#Get relevent columns to encode
columnsToEnc=['Sex', 'Title', 'AgeBin', 'FareBin', "Embarked"]

#Encode
label = LabelEncoder()
data1['Sex_Code'] = label.fit_transform(data1['Sex'])
data1['Embarked_Code'] = label.fit_transform(data1['Embarked'])
data1['Title_Code'] = label.fit_transform(data1['Title'])
data1['AgeBin_Code'] = label.fit_transform(data1['AgeBin'])
data1['FareBin_Code'] = label.fit_transform(data1['FareBin'])

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




#Split Training and Testing Data#
#-------------------------------#


train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc] ,data1[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)



#ML part#
#-------#

MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

]

#split dataset in cross-validation
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = data1[Target]

row_index = 0
for alg in MLA:
    # set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv=cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3  # let's know the worst that can happen!

    # save MLA predictions - see section 6 for usage
    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])

    row_index += 1

# print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)
MLA_compare
# MLA_predict






















"""
array = data1.values
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
#data1.fillna(-1, inplace = True)


# Encode whole data1 without NA values but keep data1 whole
# Pass TF only data1 without NA values


"""



"""
#Split Cabin column into floor and rooms columns. i.e "C130 C230" ---> C - 130 - 120
newframe = data1.Cabin.str.split(" ", expand=True)
data1.Cabin= data1.Cabin.str.extract('(\w)')
newframe = newframe.apply(lambda x: x.str.extract('(\d+)'))
data1 = pd.concat([data1, newframe],axis=1)

data1.rename(columns={0: 'cabin#1',1: 'cabin#2',2: 'cabin#3',3: 'cabin#4'}, inplace = True)

#Groupby Ticket and then give each person the count of the group
grp = data1.groupby("Ticket").size()  #could have also used data1.Ticket.value_counts()
data1['SameTicket']= data1.Ticket.map(grp)

#Make a  FamilyName column
data1["FamilyName"]= data1.Name.str.extract('^(\w+)')

#Give group numbers for each group of same ticket
data1["TicketGrp"]= pd.Categorical(data1.Ticket).codes

data1[labels] = data1[labels].apply(lambda x: d[x.name].fit_transform(x))
"""