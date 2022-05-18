#!/usr/bin/env python
# coding: utf-8

# # _Nuwe JOBarcelona Hackathon Qualifying Challenge_
#     _20220518, Kyla Juett_

# ## Business Case
#     El dataset de clientes 'train.csv' contiene las siguientes variables:
#         Hour: Hora a la que se ha hecho la medición.
#         Minutes: Minutos en los que se ha realizado la medición.
#         Sensor_alpha: Posición del insecto al sensor alpha.
#         Sensor_beta: Posición del insecto al sensor beta.
#         Sensor_gamma: Posición del insecto al sensor gamma.
#         Sensor_alpha_plus: Posición del insecto al sensor alpha+.
#         Sensor_beta_plus: Posición del insecto al sensor beta+.
#         Sensor_gamma_plus: Posición del insecto al sensor gamma+.
#         Insect: Categoría de insecto.
#             0 -> Lepidoptero
#             1 -> Himenoptera
#             2 -> Diptera


# ## Getting Started

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px

import pandas_profiling
from pandas_profiling.report.presentation.core import Alerts
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold

import warnings
warnings.filterwarnings('ignore')



## import functions
#import sys
#sys.path.insert(0, '/Users/kylajuett/Desktop/allWomen/0 bootcamp/functions' )
#from fx_MLClassification import *
#from fx_Pipeline_Classification import *



# load the train dataset
df = pd.read_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/train.csv', index_col=0) 


# and test
data = pd.read_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/test_x.csv', index_col=0) 


# ## Functions
#     copied from imported files so it works elsewhere



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier




def C_metrics_train(model, X_train, y_train):
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
    
    scores = cross_validate(model, X_train, y_train, cv=10, scoring=scoring)
    ypredTrain = model.predict(X_train)
    Acc_train = scores['test_acc'].mean()
    Precision_train = scores['test_prec_macro'].mean()
    Recall_train = scores['test_rec_macro'].mean()
    F1_train = scores['test_f1_macro'].mean()
    conf_matrix_train = confusion_matrix(y_train, ypredTrain)
    from sklearn.metrics import classification_report
    statist_train = []
   
    list_metrics = [Acc_train, Precision_train, Recall_train, F1_train]
    statist_train.append(list_metrics)
    statist_train = pd.DataFrame(statist_train,columns = ['Accuracy', 'Precision', 'Recall', 'F1'], index = ['Train'])
    
    print('-----------------------------------------')
    print('TRAIN results')
    print('-----------------------------------------')
    print('Confusion Matrix \n', conf_matrix_train)
    return statist_train


def C_metrics_test(model, X_test, y_test):
    
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
    
    scores = cross_validate(model, X_test, y_test, cv=10, scoring=scoring)
    ypredtest = model.predict(X_test)
    report = classification_report(y_test, ypredtest,zero_division=0, output_dict=True)
    report = pd.DataFrame(report).T
    
    Acc_test = report.loc['accuracy', :].mean()  
    Rest_metrics = report.iloc[:-3,:]
    
    Precision_test = Rest_metrics.loc[:,'precision'].mean()
    Recall_test = Rest_metrics.loc[:,'recall'].mean()
    F1_test = Rest_metrics.loc[:,'f1-score'].mean()
    conf_matrix_test = confusion_matrix(y_test, ypredtest)
    
    statist_test = []
   
    list_metrics = [Acc_test, Precision_test, Recall_test, F1_test]
    statist_test.append(list_metrics)
    statist_test = pd.DataFrame(statist_test, columns = ['Accuracy', 'Precision', 'Recall', 'F1'], index = ['test'])
     
    print('-----------------------------------------')
    print('TEST results')
    print('-----------------------------------------')
    print('Confusion Matrix \n', conf_matrix_test)
    print(' Classification Report \n', Rest_metrics)
    return statist_test


def C_Allmetrics(model, X_train, y_train, X_test, y_test):
    
    stats_train = C_metrics_train(model, X_train, y_train)
    stats_test = C_metrics_test(model, X_test, y_test)
    final_metrics = pd.concat([stats_train, stats_test])
    print()
    print('++++++++ Summary of the Metrics ++++++++')
    print(final_metrics)
    return final_metrics


def GetBasedModels():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('SVM'  , SVC()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    #basedModels.append(('ET'   , ExtraTreesClassifier())) 
    #basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    #basedModels.append(('NB'   , GaussianNB()))
    #basedModels.append(('AB'   , AdaBoostClassifier()))
    #basedModels.append(('GBM'  , GradientBoostingClassifier()))
    return basedModels


def BasedModels(X_train, y_train, scoring, models):
    """
    BasedModels will return the evaluation metric 'AUC' after performing
    a CV for each of the models
    input:
    X_train, y_train, scoring, models
    models = array containing the different models previously instantiated 
    
    output:
    names = names of the diff models tested
    results = results of the diff models
    """

    num_folds = 10
    scoring = scoring
    results = []
    names = []
    
    for name, model in models:
        cv_results = cross_val_score(model, X_train,
                                     y_train, cv=num_folds, scoring=scoring)
        results.append(cv_results.mean())
        names.append(name)
        msg = "%s: %s = %f (std = %f)" % (name, scoring,
                                                cv_results.mean(), 
                                                cv_results.std())
        print(msg)
    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': results})    
    return scoreDataFrame



def GetScaledModel(nameOfScaler):
    """
    arg:
    nameOfScaler = 'standard' (standardize),  'minmax', or 'robustscaler'
    """
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
    elif nameOfScaler == 'robustscaler':
        scaler = RobustScaler()

    pipelines = []
    pipelines.append((nameOfScaler+'LR', 
                      Pipeline([('Scaler', scaler),
                                ('LR', LogisticRegression())])))
    
    pipelines.append((nameOfScaler+'KNN', 
                      Pipeline([('Scaler', scaler),('KNN', 
                                                   KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART', 
                      Pipeline([('Scaler', scaler),
                                ('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfScaler+'SVM',
                      Pipeline([('Scaler', scaler),
                                ('SVM', SVC(kernel = 'rbf'))])))
    pipelines.append((nameOfScaler+'RF', 
                      Pipeline([('Scaler', scaler),
                                ('RF', RandomForestClassifier())])))
    
    #pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier())])  ))
    #pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    #pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    #pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    #pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))

    return pipelines 




# ## EDA


print(df.duplicated().sum()) # none
print(df.info()) # so clean! <heart eyes>
df.head(-10)


df.Insect.value_counts()
# UNbalanced: 0    3519, 1    2793, 2     689
## CONSIDER DROPPING CLASS 2 

print(df.Insect.value_counts())
print(df.Insect.value_counts(normalize=True))


prof = df.profile_report(sort=None)
prof.to_file(output_file='profile_report_bugs.html') # all evenly distributed! <3
prof
# important correlations: 'hour', 'Sensor_alpha_plus', & 'Sensor_beta' 


# ### Data Viz: Pairplot of Train DF
g = sns.pairplot(df, hue="Insect", palette="plasma")


# ## Feature Importance

# define the target
X = df.drop(['Insect'], axis=1)
y = df['Insect']

# instantiate & fit the RF Classifier model
clf = RandomForestClassifier(random_state=37)
clf.fit(X,y)

feature_importance = clf.feature_importances_
feature_importance


# make importances relative to max importance 
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feature_importance


importance_ = pd.DataFrame(feature_importance,
                           index= X.columns)

importance_ = importance_.reset_index()
importance_.columns = ['Column_names', 'Importance_%']

importance_ = importance_.sort_values(by= 'Importance_%', ascending=False)
importance_


px.bar(importance_, x= 'Importance_%', y= 'Column_names', color= 'Column_names',orientation='h')


# ## Models!

# Train-Test Split within Train DF
X = df[['Hour', 'Sensor_alpha_plus', 'Sensor_beta', 'Sensor_gamma']]  # above 45% relative importance
y = df['Insect']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=37)    



# ### Logistic Regression

# instantiate & fit the model
clf_logreg = LogisticRegression()
clf_logreg.fit(X_train, y_train)


cv = StratifiedKFold(n_splits=10, shuffle = False)#, random_state = 37)


y_pred_class_logreg = cross_val_predict(clf_logreg, X_train, y_train, cv = cv)

y_pred_prob_logreg = cross_val_predict(clf_logreg, X_train, y_train, cv = cv, method="predict_proba")

y_pred_prob_logreg_class0 = y_pred_prob_logreg[:, 0]
y_pred_prob_logreg_class1 = y_pred_prob_logreg[:, 1]
y_pred_prob_logreg_class2 = y_pred_prob_logreg[:, 2]

print('LogReg predicted prob of the first 10 samples belonging to insect class 0\n', y_pred_prob_logreg_class0[0:10])
print('LogReg predicted prob of the first 10 samples belonging to insect class 1\n', y_pred_prob_logreg_class1[0:10])
print('LogReg predicted prob of the first 10 samples belonging to insect class 2\n', y_pred_prob_logreg_class2[0:10])
# again, ¿drop class 2?


# ### Random Forest Classifier

# instantiate & fit the model
clf_rfc = RandomForestClassifier()
clf_rfc.fit(X_train, y_train)


y_pred_class_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv)

y_pred_prob_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv, method="predict_proba")
y_pred_prob_rfc_class0 = y_pred_prob_rfc[:, 0]
y_pred_prob_rfc_class1 = y_pred_prob_rfc[:, 1]
y_pred_prob_rfc_class2 = y_pred_prob_rfc[:, 2]

print('RandomForestClas predicted prob of the first 10 samples belonging to insect class 0\n', y_pred_prob_rfc_class0[0:10])
print('RandomForestClas predicted prob of the first 10 samples belonging to insect class 1\n', y_pred_prob_rfc_class1[0:10])
print('RandomForestClas predicted prob of the first 10 samples belonging to insect class 2\n', y_pred_prob_rfc_class2[0:10])



acc_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'accuracy').mean()
acc_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'accuracy').mean()

print('\n Accuracy LogRegression:', acc_logreg)  # 74%
print('\n Accuracy RandomForest:', acc_rfc)  # 90%



f1m_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'f1_macro').mean()
f1m_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'f1_macro').mean()

print('\n F1_macro LogRegression:', f1m_logreg)  # 66%
print('\n F1_macro RandomForest:', f1m_rfc)  # 86%



logreg_matrix = metrics.confusion_matrix(y_train, y_pred_class_logreg)
print('logreg_matrix')
print(logreg_matrix) # class2 v bad



rfc_matrix = metrics.confusion_matrix(y_train, y_pred_class_rfc)
print('rfc_matrix') 
print(rfc_matrix) # class2 better, but still bad




# ## Comparison: Classification Report

report_logreg = metrics.classification_report(y_train, y_pred_class_logreg)   
report_rfc = metrics.classification_report(y_train, y_pred_class_rfc)

print("report_logreg " + "\n" + report_logreg, "report_rfc " + "\n" +  report_rfc, sep = "\n")


# # Pipeline!
#     Classification


# define baseline models
models = GetBasedModel()
models


# scoring = 'f1_macro'


Base_model = BasedModels(X_train, y_train, 'roc_auc', models) # NaNs

Base_model = BasedModels(X_train, y_train, 'accuracy', models)

Base_model = BasedModels(X_train, y_train, 'f1_macro', models) 


MetricsClas(models,X_train, y_train, X_test, y_test)
# and the winner is... Random Forest! (followed closely by KNN & SVM)




# ## Feature Scaling
#     maybe we can do something about class2?
#         (even though scaling is not usually required for trees)


Base_model


# ### Standard Scaler

models = GetScaledModel('standard')
models

scaledScoreStandard = BasedModels(X_train, y_train, 'roc_auc', models) # NaNs
scaledScoreStandard = BasedModels(X_train, y_train, 'r2', models) # eek, that's bad.
scaledScoreStandard = BasedModels(X_train, y_train, 'accuracy', models) # RF, KNN, CART/SVM
scaledScoreStandard = BasedModels(X_train, y_train, 'f1_macro', models) # RF, KNN, CART/SVM 
# IMPORTANT 


# easier viewing: concatenate the results using the F1_macro calculated above
compareModels = pd.concat([Base_model,scaledScoreStandard], axis=1)
compareModels  # very similar to original


# ### MinMax

models = GetScaledModel('minmax')
scaledScoreMinMax = BasedModels(X_train, y_train, 'f1_macro', models)

compareModels = pd.concat([Base_model, scaledScoreStandard,
                           scaledScoreMinMax], axis=1)
compareModels  # OMG, it's actually (sliiightly) worse! 🤷🏽‍♀️


# ### Robust Scaler

models = GetScaledModel('robustscaler')
scaledScoreRobustSc= BasedModels(X_train, y_train,'f1_macro', models)


compareModels = pd.concat([Base_model,scaledScoreStandard,
                           scaledScoreMinMax, scaledScoreRobustSc], axis=1)
compareModels  # 'bout the same as the original (un-scaled)


# # Winning Model: Unscaled RF
#     all metrics for the original (unscaled) Random Forest:

C_Allmetrics(clf_rfc, X_train, y_train, X_test, y_test)
# "too good to be true" Train results in Confusion Matrix, but then also good metrics for both Train & Test sets 



# # Test It
#     with the real Test df


print(data.duplicated().sum()) # none
print(data.info()) # so clean! <3
data.head(-10)



# same as above: split + train with only the Train DF
X = df.drop(['Insect'], axis=1)
y = df['Insect']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=37, stratify=y) # lots to train on (since we already know how well the model  does) 


# instantiate & fit the RF model
clf_rfc = RandomForestClassifier()
clf_rfc.fit(X_train, y_train)


# make (& view) predictions
y_pred=clf.predict(data)

print(y_pred.shape)
y_pred

print(len(X_test))
3000 - (len(X_test))



# this is definitely not a correct measurement (why?!)
C_Allmetrics(clf_rfc, X_train, y_train, X_test, y_pred[y_pred[0:(len(X_test))]])

C_Allmetrics(clf_rfc, X_train, y_train, X_test, y_pred[y_pred[(3000-len(X_test)):3000]])



# save predictions to a new column in test_x DF
data['Insect_pred'] = y_pred
data


g2 = sns.pairplot(data, hue="Insect_pred", palette="plasma")


print(accuracy_score(y_test, y_pred[y_pred[(3000-len(y_test)):3000]])) # 40.4% 😱
confusion_matrix(y_test, y_pred[y_pred[(3000-len(y_test)):3000]]) # that doesn't look very promising
print(classification_report(y_train[y_train[0:(len(y_pred))]], y_pred))



results = data.Insect_pred
results = pd.DataFrame(results)
results.to_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/results.csv')
results


# how close are the predictions (to each other)?    very, but it doesn't seem to matter...?
print('test_x actual predictions \n', data.Insect_pred.value_counts())

orig_perc = df.Insect.value_counts(normalize=True)
pred_perc = data.Insect_pred.value_counts(normalize=True)

print('\n difference, in % \n', (orig_perc - pred_perc)*100) # <1-3% differences for each class




# # Conclusion
#     So we have TERRIBLE test metrics, despite the excellent train metrics (83-86% F1, 89-90% accuracy), & regardless of the size of the train sample (I ran the model with test sizes of 10%, 20%, & 33%)
#     This is probably true for my model: "F1 is a quick way to tell whether the classifier is actually good at identifying members of a class, or if it is finding shortcuts (e.g., just identifying everything as a member of a large* class)."
#         https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56#:~:text=F1%20score%20is%20a%20little,low%2C%20F1%20will%20be%20low.
#     * The difference is, though, that with this dataset, it's the smallest class that's most inaccurate.



# # This is the End 🎶
results.to_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/results.csv') 
df.to_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/df_train.csv') 
data.to_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/data_testx.csv') 
