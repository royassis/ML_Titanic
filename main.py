from pcg.imports2 import *
import re


#Data cleaning and preprocessing#
#-------------------------------#

#Load df from file
dframe = pd.read_csv("train.csv")

#Deak with NA values

dframe['Age'].fillna(dframe['Age'].median(), inplace=True)
dframe['Embarked'].fillna(dframe['Embarked'].mode()[0], inplace = True)
dframe['Fare'].fillna(dframe['Fare'].median(), inplace = True)


#Split Cabin column into floor and rooms columns. i.e "C130 C230" ---> C - 130 - 120
newframe = dframe.Cabin.str.split(" ", expand=True)
dframe.Cabin= dframe.Cabin.str.extract('(\w)')
newframe = newframe.apply(lambda x: x.str.extract('(\d+)'))
dframe = pd.concat([dframe, newframe],axis=1)


#Groupby Ticket and then give each person the count of the group
grp = dframe.groupby("Ticket").size()  #could have also used dframe.Ticket.value_counts()
dframe['SameTicket']= dframe.Ticket.map(grp)

#Give group numbers for each group of same ticket
dframe["TicketGrp"]= pd.Categorical(dframe.Ticket).codes

#Make a column of Parch + SibSp = Family
dframe['Family'] = dframe['Parch'] + dframe['SibSp']

#Make a column is alone
dframe['Alone'] = dframe['Family'].dropna().apply(lambda x: 1 if x == 0 else 0)

#Make a Title column
dframe["Title"]= dframe.Name.str.extract('^.*?,\s(.*?)\.')
#CLean Title
validTitles = (dframe["Title"].value_counts()/dframe["Title"].shape[0])<0.1
dframe["Title"] = dframe["Title"].apply(lambda x: "Misc" if validTitles.loc[x] == True else x )

#Make a  FamilyName column
dframe["FamilyName"]= dframe.Name.str.extract('^(\w+)')

#Binning relevent columns
dframe['AgeBin'] = pd.cut(dframe['Age'].astype(int), 5)
dframe['FareBin'] = pd.qcut(dframe['Fare'], 4)

#Encoding
label = LabelEncoder()
columnsToEnc=['Sex', 'Title', 'AgeBin', 'FareBin']
for i in columnsToEnc:
    dframe[i.__str__()+"_code"]=label.fit_transform(dframe[i])

#Turn NA values to -1 in order to be used in learning algorithms
#dframe.fillna(-1, inplace = True)


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


