#!/usr/bin/env python3
print('Hello world')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
#####load files into variables
train = pd.read_csv("LoanPredictionTrain.csv")
test = pd.read_csv("LoanPredictionTrain copy.csv")

train_original = train.copy()
test_original = test.copy()
print (train.columns) 
print (test.columns)
print (train.dtypes)



print(train.shape, test.shape)
##display a distinct count of values
print(train['Loan_Status'].value_counts())

print(train['Dependents'].value_counts())

#set normalize to true to print proportions instead of values
print(train['Loan_Status'].value_counts(normalize=True))
train['Loan_Status'].value_counts(normalize=True)
train['Loan_Status'].value_counts().plot.bar()
plt.show()

plt.figure("visualize categorical features")
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title='Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit_Histsory')


plt.figure("Visualize ordinal features")
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title='Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title='Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title='Property_Area')
plt.show()


plt.figure("Distribution of applicant income")
plt.subplot(121)
sns.distplot(train['ApplicantIncome'])

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()


train.boxplot(column='ApplicantIncome', by='Education')
plt.suptitle("See if there is a relationship between Applicant income and education level")
plt.show()

plt.figure("Co-applicant distribution")
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))
plt.show()

plt.figure("Distribution of LoanAmount variable")
plt.subplot(121)
df=train.dropna()
sns.distplot(train['LoanAmount']);
plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))
plt.show()

Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, title="Relation of categorical Gender value and Loan Status",figsize=(4,4))
plt.show()

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar",title="Relation of categorical value of Married(Y/N) and Loan Status", stacked=True,figsize=(4,4))
plt.show()

Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar",title="Relation of categorical value of number of Dependents and Loan Status",stacked=True)
plt.show()
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", title="Relation of categorical value of Education level and Loan Status",stacked=True,figsize=(4,4))
plt.show()
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar",title="Relation of categorical value of Self-employed(Y/N) and Loan Status",stacked=True, figsize=(4,4))
plt.show()

Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True, title="Relation of categorical value of Qualified credit history and Loan Status",figsize=(4,4))
plt.show()

Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar",title="Relation of ordinal value of property area and Loan Status",stacked=True)
plt.show()

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar(title="")
plt.show()

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",title="Relation of numberical value applicant income with target variable",stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')

bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",title="Relation of numberical value co-applicant income with target variable", stacked=True)
plt.xlabel('CoapplicantIncome')
P = plt.ylabel('Percentage')


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",title="Relation of numberical value applicant plus co-applicant income with target variable",stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')

bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar",title="Relation of numberical value loan amout with target variable",stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')
plt.show()

train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin','Total_Income_bin', 'Total_Income'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)
matrix = train.corr() 
ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix,vmax=.8, square=True, cmap="BuPu");
plt.show()
print("count missing values")
print(train.isnull().sum())
#fill least amount of missing values with mode because they are categorical values
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
print("value count of the loan_amount_term variable")
print(train['Loan_Amount_Term'].value_counts())
##360 is the mode so it will be used to fill in the missing values
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

##replace missing numerical values with mean or median(if outliers)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

#outliers cause the graphs to be left or righ skewed, taking a log transformation to varibles with
##outliers create a normal distribution because the log transformation does not
##affect the smaller values
plt.figure("Example of using log function in the LoanAmount to normalize data with outliers")
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])
plt.show()
####

#model 
##drop loan_ID so that it does not have an effect on loan status
train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)
X = train.drop('Loan_Status',1)
y = train.Loan_Status

X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100,
                   multi_class='ovr', n_jobs=1,penalty='l2', random_state=1, solver='liblinear', tol=0.0001, verbose=0,warm_start=False)

pred_cv = model.predict(x_cv)
print(accuracy_score(y_cv,pred_cv))

pred_test = model.predict(test)

submission=pd.read_csv("LoanPredictionTrain Rand.csv")

submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']

submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')



from sklearn.model_selection import StratifiedKFold

i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold{} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model =LogisticRegression(random_state=1)
    model.fit(xtr, ytr)
    pred_test =model.predict(xvl)
    score = accuracy_score(yvl,pred_test)
    print('accuracy_score',score)
    i+=1
pred_test = model.predict(test)
pred=model.predict_proba(xvl)[:,1]

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(yvl, pred)
auc =metrics.roc_auc_score(yvl, pred)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
##plt.show()

submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Logistic.csv')
plt.show()


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
sns.distplot(train['Total_Income'])


train['Total_Income_log'] = np.log(train['Total_Income'])
sns.distplot(train['Total_Income_log'])
test['Total_Income_log'] = np.log(test['Total_Income'])


train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']
sns.distplot(train['EMI'])


train['Balance Income']=train['Total_Income']-(train['EMI']*1000)
# Multiply with 1000 to make the units equal
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)
sns.distplot(train['Balance Income'])

train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
'Loan_Amount_Term'], axis=1)
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
'Loan_Amount_Term'], axis=1)



#decision tree, decision tree model with 5 folds of cross validation.

from sklearn import tree
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):    
    print('\n{} of kfold {}'.format(i,kf.n_splits))    
    xtr,xvl = X.loc[train_index],X.loc[test_index]    
    ytr,yvl = y[train_index],y[test_index]        
    model = tree.DecisionTreeClassifier(random_state=1)    
    model.fit(xtr, ytr)    
    pred_test = model.predict(xvl)    
    score = accuracy_score(yvl,pred_test)    
    print('accuracy_score',score)    
    i+=1
pred_test = model.predict(test)
plt.show()
