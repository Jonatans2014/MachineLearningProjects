
# coding: utf-8

# Created on Wed Jul 25 11:31:16 2018
# 
# Jonatans Almeida De Souza
# 
# 
# 
# 
# Kaggle Titanic Disaster
# 
# Description
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# 
# 
# Goal
# The Goal is to predict if a passenger survived the sinking of the Titanic or not. 
# For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.

#check for missing V
"""
print('Train columns with null values:\n', data1.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-"*10)

data_raw.describe(include = 'all')"""


#try this     #complete embarked with mode
    #dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)


# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re as re
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')
# In[2]:


#load train and test dataset
train = pd.read_csv('Titanic_train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})

#asssign the PassengId to passId VAR for the test prediction
passiD= test.PassengerId

#assign the whole dataset to full_data
full_data = [train, test]


# In[3]:


train.head(10)


# Feature engineering
# 

# In[4]:


"""check the impact of pclass collumn on the dataset. 
   It is noticable that 62% of people on the first class survived.
"""
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[5]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()


# In[6]:


#checking if SibSp would have any impact on the prediction Sibsp stands for Siblings/Spouse
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()


# In[7]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()


# It`s noticeable that people who were with someone else wether siblings, spouse, parents or children, survived more than people who were alone 
# 
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

"""Only Fare feature seems to have a significative correlation with the survival probability.

It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features"""


g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")



g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45) 
"""There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories."""
# In[8]:


#join these two collums Sibsp and Parch 
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] +1
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()


# In[31]:


#Create a new collumn IsAlone to check if the person is alone on the titanic or not
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


#getName
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)




# In[10]:


#check if Age has missing value and fill it with the mean()
train.Age.isnull().values.any()



#split titles

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# In[11]:


print(pd.crosstab(train['Title'], train['Sex']))

#fill the missig values in age with the mean and make it an integer
for dataset in full_data:
    dataset['Age'].fillna((dataset['Age'].mean()), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['Age'].head(20)
    
    #Fill missing values of Fare
    dataset['Fare'].fillna((dataset['Fare'].mean()), inplace=True)
    dataset['Fare'].head(20)
    
     
    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

test.Age.isnull().values.any()
train.Age.isnull().values.any()



# In[12]:



 


# In[13]:




# Might categorise the age Collumn
# 

# In[14]:


#check who survied the most women or men
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()



# Now we should clean our data and tranform categorical variable into numerical

# In[15]:

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#transform Sex Categ in Numerical
for dataset in full_data:
    # Mapping Sex
    #dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
     dataset['Sex'] = labelencoder_X.fit_transform(dataset['Sex'])
     # Mapping Sex
     dataset['Embarked'] = labelencoder_X.fit_transform(dataset['Embarked'])
    
     dataset['Title'] = labelencoder_X.fit_transform(dataset['Title'])




# In[16]:


train.head()


# In[17]:


#drop few features which are dirty and may not contribute to the prediction at all


# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',
                 'Parch', 'FamilySize']
#drop all collumns from above in train and test dataset
train = train.drop(drop_elements, axis=1)
test = test.drop(drop_elements, axis=1)


# In[18]:



train.head(10)


# In[19]:


test.head(10)


# In[20]:


#split the survivide collumn which will use to trian our model as the Y value
y = train.pop('Survived')


# In[21]:


train.head(10).info()


# In[22]:


train = train.values
test  = test.values



from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel



from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline




classifiers = [
        
        
          #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    LinearSVC(),
      Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC())),
  ('classification', RandomForestClassifier(criterion= 'entropy', max_depth= 6, n_estimators= 100, oob_score= True,n_jobs=-1, random_state= 0))
     ])
    
    
    ]

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train
y = y

acc_dict = {}

for train_index, test_index in sss.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	for clf in classifiers:
		name = clf.__class__.__name__
		clf.fit(X_train, y_train)
		train_predictions = clf.predict(X_test)
		acc = accuracy_score(y_test, train_predictions)
		if name in acc_dict:
			acc_dict[name] += acc
		else:
			acc_dict[name] = acc

for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf] / 10.0
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)

log
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


# In[134]:


#misc libraries
import random
import time


#why choose one model, when you can pick them all with voting classifier
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
#removed models w/o attribute 'predict_proba' required for vote classifier and models with a 1.0 correlation to another model
vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc',ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier())
]


#Hard Vote or majority rules
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, X_train, y_train, cv  = sss)
vote_hard.fit(X_train, y_train)

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)




# In[135]:


#Soft Vote or weighted probabilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft,  X_train, y_train, cv  = sss)
vote_soft.fit( X_train, y_train)

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)



# In[135]:



import random
import time


#WARNING: Running is very computational intensive and time expensive.
#Code is written for experimental/developmental purposes and not production ready!

    
#Hyperparameter Tune with GridSearchCV: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]


grid_param = [
            [{
            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
            'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn, #default=1
            #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R
            'random_state': grid_seed
            }],
       
    
            [{
            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
            'n_estimators': grid_n_estimator, #default=10
            'max_samples': grid_ratio, #default=1.0
            'random_state': grid_seed
             }],

    
            [{
            #ExtraTreesClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'random_state': grid_seed
             }],


            [{
            #GradientBoostingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            #'loss': ['deviance', 'exponential'], #default=’deviance’
            'learning_rate': grid_learn, #default=0.1 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            'n_estimators': grid_n_estimator, #default=100 -- 12/31/17 set to reduce runtime -- The best parameter for GradientBoostingClassifier is {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 300, 'random_state': 0} with a runtime of 264.45 seconds.
            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': grid_max_depth, #default=3   
            'random_state': grid_seed
             }],

    
            [{
            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            'n_estimators': grid_n_estimator, #default=10
            'criterion': grid_criterion, #default=”gini”
            'max_depth': grid_max_depth, #default=None
            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime -- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.
            'random_state': grid_seed
             }]
   
        ]



start_total = time.perf_counter() #https://docs.python.org/3/library/time.html#time.perf_counter
for clf, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

    #print(clf[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
    #print(param)
    
    
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = sss, scoring = 'roc_auc')
    best_search.fit( X_train, y_train)
    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param) 


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-'*10)




# In[137]:







#submit file
#submit = dataset[['PassengerId','Survived']]
#submit.to_csv("submit.csv", index=False)


#train using ramdon forest Classifier

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier( n_estimators = 100,criterion= 'entropy', max_depth= 6, oob_score= True,random_state = 0  )
classifier.fit(train, y)

"""
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(learning_rate= 0.05, n_estimators= 300, max_depth= 2,  random_state = 0  )
classifier.fit(train, y)
"""

"""
from sklearn.ensemble import BaggingClassifier
classifier = BaggingClassifier(n_estimators= 300,max_samples = 0.25,  random_state = 0  )
classifier.fit(train, y)"""

# Applying k-Fold Cross Validation



vote_est = [
    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html

    ('bc', ensemble.BaggingClassifier(n_estimators= 300,max_samples = 0.25,  random_state = 0  )),
    ('gbc', ensemble.GradientBoostingClassifier(learning_rate= 0.05, n_estimators= 300, max_depth= 2,  random_state = 0  )),
    ('rfc', ensemble.RandomForestClassifier( n_estimators = 100,criterion= 'entropy', max_depth= 6, oob_score= True,random_state = 0  ))
]



#Soft Vote or weighted probabilities
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_soft_cv = model_selection.cross_validate(vote_soft,  X_train, y_train, cv  = sss)
vote_soft.fit( X_train, y_train)

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)




result = vote_soft.predict(test)





predict = pd.DataFrame()


#predict['Survived'] = result

predict['PassengerId'] = dataset.PassengerId
predict['Survived'] = result
predict.to_csv('TitanicRandForestPred.csv', index = False)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = train, y = y, cv = 10)
accuracies.mean()




#submit = dataset[['PassengerId','Survived']]
#submit.to_csv("submit.csv", index=False)

