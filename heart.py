# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:24:44 2022

@author: AMD
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#%%

#%%
import scipy.stats as ss
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% STATIC
DATA_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')

#%% STEP 1 DATA LOADING
df = pd.read_csv(DATA_PATH)

#%% STEP 2 DATA INSPECTION
info = df.describe().T
df.info()

plt.figure(figsize=(20,20))
df.boxplot()
plt.show()


# CHECK NaNs VALUE
df.isna().sum() #NO NaNs value in the dataset

# CHECK DUPLICATE VALUE
df.duplicated().sum()
df[df.duplicated()]

# show the distribution of the data
con_columns = ['age','trtbps','chol','thalachh','oldpeak']
cat_columns = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']

for con in con_columns:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

for cat in cat_columns:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()
    

#%% STEP 3 Data Cleaning

# 1) Remove duplicated Data
df = df.drop_duplicates()
print(df.duplicated().sum())
print(df[df.duplicated()])

# 2) REMOVE NANS
# NO NEED BECAUSE NO NANS IN THIS DATASETS
# Nothing More To Clean

#%% STEP 4 Features selection

# CON VS CAT(USE LOGISTIC REGRESSION)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

for con in con_columns:
    print(con)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis =-1),df['output'])
    print(lr.score(np.expand_dims(df[con],axis = -1),df['output']))

# since age, trtbps,thalachh(maximum heart rate achieved)and oldpeak(Previous peak) achieved 61% ,57%, 70% and 68%
# thus, those features features will be selected for subsequent step

# CAT VS CAT(USE CREMERS'V)

for cat in cat_columns:
    print(cat)
    confussion_mat = pd.crosstab(df[cat],df['output']).to_numpy()
    print(cramers_corrected_stat(confussion_mat))

# since thatll has 0.521 , cp( Chest Pain type chest pain type) has 0.508 , caa has 0.481 and exng has 0.4253 correlation
# thus, those features features will be selected for subsequent step

#%% STEP 5 Preprocessing

X = df.loc[:,['age','cp','trtbps','thalachh','exng','oldpeak','caa','thall']]

y = df['output']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state = 64)


#%% PIPELINE

#steps for standar scaler 
step_ss_logistic = [('Standard Scaler', StandardScaler()),
                    ('Logistic Classfier', LogisticRegression())]


step_ss_decision = [('Standard Scaler', StandardScaler()),
                    ('Decision Tree Classfier', DecisionTreeClassifier())]

step_ss_randomforest = [('Standard Scaler', StandardScaler()),
                        ('Random Forest Classfier', RandomForestClassifier())]

step_ss_KNN = [('Standard Scaler', StandardScaler()),
               ('Random Forest Classfier', KNeighborsClassifier())]

#step_ss_SVC = [('Standard Scaler', StandardScaler()),
#               ('Random Forest Classfier', SVC())]

#steps for MinMax scaler

step_mms_logistic = [('Min Max Scaler', MinMaxScaler()),
                     ('Logistic Classfier', LogisticRegression())]


step_mms_decision = [('Min Max Scaler', MinMaxScaler()),
                     ('Decision Tree Classfier', DecisionTreeClassifier())]

step_mms_randomforest = [('Min Max Scaler', MinMaxScaler()),
                         ('Random Forest Classfier', RandomForestClassifier())]

step_mms_KNN = [('Standard Scaler', MinMaxScaler()),
                ('Random Forest Classfier', KNeighborsClassifier())]

#step_mms_SVC = [('Standard Scaler', MinMaxScaler()),
#                ('Random Forest Classfier', SVC())]

#to create pipeline
pipeline_ss_logistic = Pipeline(step_ss_logistic)
pipeline_ss_decision = Pipeline(step_ss_decision)
pipeline_ss_randomforest = Pipeline(step_ss_randomforest)
pipeline_ss_KNN = Pipeline(step_ss_KNN)
#pipeline_ss_SVC = Pipeline(step_ss_SVC)

pipeline_mms_logistic = Pipeline(step_mms_logistic)
pipeline_mms_decision = Pipeline(step_mms_decision)
pipeline_mms_randomforest = Pipeline(step_mms_randomforest)
pipeline_mms_KNN = Pipeline(step_mms_KNN)
#pipeline_mms_SVC = Pipeline(step_mms_SVC)

pipelines = [pipeline_ss_logistic,pipeline_ss_decision,pipeline_ss_randomforest,pipeline_ss_KNN,
             pipeline_mms_logistic,pipeline_mms_decision,pipeline_mms_randomforest,pipeline_mms_KNN]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

best_accuracy = 0
pipeline_scored = []

for i, pipeline in enumerate(pipelines):
    print(pipeline.score(X_test,y_test))
    pipeline_scored.append(pipeline.score(X_test,y_test))

best_pipeline = pipelines[np.argmax(pipeline_scored)]
best_accuracy = pipeline_scored[np.argmax(pipeline_scored)]
print('The best combination of the pipeline is {} with accuracy of {}'
      .format(best_pipeline.steps,best_accuracy))


#%% STEP for SCV scalar approach (GRID SEARCH)

#fine tuning

step_ss_rf =[('Standard Scaler', StandardScaler()),
             ('RandomForestClassifier',RandomForestClassifier())]

pipeline_mms_rf = Pipeline(step_ss_rf)

grid_param = [{'RandomForestClassifier__n_estimators':[10,100,1000],
               'RandomForestClassifier__max_depth':[3,5,7,10,None],
               'RandomForestClassifier__min_samples_leaf':np.arange(1,5)}]#make sure no space there


gridsearch = GridSearchCV(pipeline_mms_rf,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model = gridsearch.fit(X_train,y_train)
best_model.score(X_test,y_test)


#%%
step_ss_rf =[('Standard Scaler', StandardScaler()),
             ('RandomForestClassifier',RandomForestClassifier(n_estimators=10,min_samples_leaf=4,max_depth=100))]

pipeline_ss_rf = Pipeline(step_ss_rf)

pipeline_ss_rf.fit(X_train,y_train)

BEST_MODEl_PATH = os.path.join(os.getcwd(),'model','best_model.pkl')
with open(BEST_MODEl_PATH,'wb') as file:
    pickle.dump(pipeline_ss_rf,file)


#%%

print(best_model.best_index_)
print(best_model.best_params_)

PKL_FNAME = os.path.join(os.getcwd(),'model','best_pipeline.pkl')

with open(PKL_FNAME,'wb') as file:
    pickle.dump(best_model,file)


#%% MODEL ANALYSIS

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
# this model is classification, use above

y_true = y_test
y_pred = best_model.predict(X_test)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print("the Accuracy score of the model is", accuracy_score(y_true, y_pred))









