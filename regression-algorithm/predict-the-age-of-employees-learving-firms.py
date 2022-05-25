import pandas as pd

df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.insert(len(df.columns) - 1, 'Age', df.pop('Age'))

df.head(10)
print(df.shape)
df.info()
missing_value = df.isna().sum()
print(missing_value)
df.describe().T
df.skew()

# univariate 

import matplotlib.pyplot as plt
from matplotlib import pyplot

df.hist()
plt.gcf().set_size_inches(20,20)
plt.show()

df.plot(kind = 'box', subplots = True, layout=(6, 5), sharex = False)
plt.gcf().set_size_inches(20, 20)
pyplot.show()

# correlation

import seaborn as sns

# df = df.drop(columns =['EmployeeCount'])
# df = df.drop(columns =['StandardHours'])

plt.figure(figsize=(18,10))
sns.heatmap(df.corr(), annot= True, fmt='.1g')

#converting categotical features to numeric using LabelEncoder() for data transformation

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data = df
data['Attrition'] = le.fit_transform(data['Attrition'])
data['BusinessTravel'] = le.fit_transform(data['BusinessTravel'])
data['Department'] = le.fit_transform(data['Department'])
data['EducationField'] = le.fit_transform(data['EducationField'])
data['Gender'] = le.fit_transform(data['Gender'])
data['JobRole'] = le.fit_transform(data['JobRole'])
data['MaritalStatus'] = le.fit_transform(data['MaritalStatus'])
data['Over18'] = le.fit_transform(data['Over18'])
data['OverTime'] = le.fit_transform(data['OverTime'])

data.head()

# data transformation using scikit learn MinMaxScaler to Rescale

from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

filename = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

array = data.values
x = array[:, : 34]
y = array[:, 34]

model = MinMaxScaler(feature_range= (0, 1))
rescaledx = model.fit_transform(x)

set_printoptions(precision= 3)
print(rescaledx[0:3, :])

rescaledx_data = pd.DataFrame(rescaledx)
rescaledx_data.columns = ['Attrition', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

rescaledx_data['Age'] = data['Age']
rescaledx_data.head()

rescaledx_data.describe().T

# feature selection using RFE

from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

filename = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
array = rescaledx_data.values
x = array[:, : 34]
y = array[:, 34]

model = LinearRegression()
rfe = RFE(model,25)
fit = rfe.fit(x, y)

print('Num features: {}'.format(fit.n_features_))
print('selection feature: {}'.format(fit.support_))
print('feature ranking: {}'.format(fit.ranking_))

cols =['Attrition', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

ranking = list(fit.ranking_)
feature_lt = []

for col, ran in zip(cols, ranking):
    if ran == 1:
        feature_lt.append(col)

cols =['Attrition', 'BusinessTravel', 'DailyRate', 'Department',
       'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
       'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
       'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
       'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']

ranking = list(fit.ranking_)
feature_lt = []

for col, ran in zip(cols, ranking):
    if ran == 1:
        feature_lt.append(col)

dframe = rescaledx_data.loc[:, ['Attrition','DailyRate','Department','Education',
                                'EducationField','HourlyRate','JobInvolvement','JobLevel','JobRole',
                                'JobSatisfaction', 'MaritalStatus','MonthlyIncome','MonthlyRate',
                                'NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating',
                                'RelationshipSatisfaction','TotalWorkingYears','TrainingTimesLastYear',
                                'WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion',
                                'YearsWithCurrManager', 'Age']]

import numpy as np

matrix = np.triu(dframe.corr())
plt.figure(figsize = (18, 12))
sns.heatmap(dframe.corr(), annot = True, mask = matrix, fmt='.1g')

# Regression Algorithms

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Linear MLA for Regression

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


# Non-Linear MLA for Regression

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# automate algorithms

algorithms = {'LinearRegression':LinearRegression, 'Ridge':Ridge, 'Lasso':Lasso, 'ElasticNet':ElasticNet,
              'KNNR':KNeighborsRegressor, 'DecisionTreeRegressor':DecisionTreeRegressor, 'SVR':SVR}

def res(x, y, alg, cv):
    model = alg()
    resp = cross_val_score(model, x, y, cv=cv)
    return resp
    
array = dframe.values
x, y = array[:, : 20], array[:, 20]

cv = KFold(n_splits= 10)

algoResList = []
for name, algos in algorithms.items():
    result = res(x, y, algos, cv)
    output = [name, np.mean(result)]
    algoResList.append(output)

algoResults = pd.DataFrame(algoResList, columns = ['Algorithm', 'Score'])





