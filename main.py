from pcg.imports2 import *
import re


#Data cleaning and preprocessing#
#-------------------------------#

#Load df from file
dframe = pd.read_csv("train.csv")

#Deal with NA values
dframe['Age'].fillna(dframe['Age'].median(), inplace=True)
dframe['Embarked'].fillna(dframe['Embarked'].mode()[0], inplace = True)
dframe['Fare'].fillna(dframe['Fare'].median(), inplace = True)

drop_column = ['PassengerId','Cabin', 'Ticket']
dframe.drop(drop_column, axis=1, inplace = True)


"""
#Split Cabin column into floor and rooms columns. i.e "C130 C230" ---> C - 130 - 120
newframe = dframe.Cabin.str.split(" ", expand=True)
dframe.Cabin= dframe.Cabin.str.extract('(\w)')
newframe = newframe.apply(lambda x: x.str.extract('(\d+)'))
dframe = pd.concat([dframe, newframe],axis=1)
"""

"""
#Groupby Ticket and then give each person the count of the group
grp = dframe.groupby("Ticket").size()  #could have also used dframe.Ticket.value_counts()
dframe['SameTicket']= dframe.Ticket.map(grp)
"""

"""
#Give numbers for each group of same ticket
dframe["TicketGrp"]= pd.Categorical(dframe.Ticket, ordered=True).codes
"""

#Make a column of Parch + SibSp = Family
dframe['FamilySize'] = dframe['Parch'] + dframe['SibSp'] +1

#Make a column is alone
dframe['IsAlone'] = dframe['FamilySize'].dropna().apply(lambda x: 1 if x == 1 else 0)

#Make a Title column
dframe["Title"]= dframe.Name.str.extract('^.*?,\s(.*?)\.')
#CLean Title
validTitles = (dframe["Title"].value_counts()/dframe["Title"].shape[0])<0.1
dframe["Title"] = dframe["Title"].apply(lambda x: "Misc" if validTitles.loc[x] == True else x )

"""
#Make a FamilyName column
dframe["FamilyName"]= dframe.Name.str.extract('^(\w+)')
"""

#Binning relevent columns
dframe['AgeBin'] = pd.cut(dframe['Age'].astype(int), 5)
dframe['FareBin'] = pd.qcut(dframe['Fare'], 4)

#Encoding
label = LabelEncoder()
columnsToEnc=['Sex', 'Title', 'AgeBin', 'FareBin', "Embarked"]
for i in columnsToEnc:
    dframe[i.__str__()+"_Code"]=label.fit_transform(dframe[i])

#Turn NA values to -1 in order to be used in learning algorithms
#dframe.fillna(-1, inplace = True)

Target  = ['Survived']
dframe_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
dframe_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']
dframe_xy = Target + dframe_x


dframe_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
dframe_xy_bin = Target + dframe_x_bin

dframe_dummy = pd.get_dummies(dframe[dframe_x])
dframe_x_dummy = dframe_dummy.columns.tolist()
dframe_x_bin_xy_dummy = Target + dframe_x_dummy


#Split Training and Testing Data#
#-------------------------------#


train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(dframe[dframe_x_calc] ,dframe[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(dframe[dframe_x_bin], dframe[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(dframe_dummy[dframe_x_dummy], dframe[Target], random_state = 0)



data1= dframe.copy(deep= True)
#Viz#
#---#

#1
plt.figure(figsize=[16,12])

plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']],
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']],
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']],
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Family Size Histogram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()



"""
#Maching learning#
#----------------#

# Split-out validation dataset
array = dframe.values
X = array[:,1:7]
Y = array[:,0]
X,Y = X,Y.astype(float)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


for x,y in enumerate(dframe.columns):
	print (x,y,dframe.iloc[x,x])

"""


"""
#Homemade functions#
#------------------#


Could have just used: dframe.isnull().sum()/dframe.shape[0]

def NullColumnPrec(dframe):
	for i in dframe.columns:
		print (i, dframe[dframe[i].isnull()].shape[0]/dframe.shape[0]) 


Could have just used: dframe.nunique()/dframe.shape[0]

def UniqueColumnPrec(dframe):
	arr = []
	for i in dframe.columns:
		number = dframe[i].unique().shape[0] / dframe.shape[0]
		arr.append([i,number])
	return arr

	for i in arr :
		print(i, "{:.2f}".format(i[0]))

#Does not work, work in progress
def text2number(dframe):
	columns = dframe.dtypes[dframe.dtypes == object].index
	dict = {}
	arr=[]
	for i in columns:
		for j in dframe[i].unique():
			arr.append(j)

	for i, j in enumerate(arr):
		dict[j] =i
	return  dict 
	"""


