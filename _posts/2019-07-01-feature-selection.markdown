---
layout: post
title:  "Feature Selection"
date:   2019-07-01 09
categories: More
---
<br />
<h4>Readings</h4>
<a href="https://machinelearningmastery.com/feature-selection-machine-learning-python/">
https://machinelearningmastery.com/feature-selection-machine-learning-python/
</a><br />
<a href="https://scikit-learn.org/stable/modules/feature_selection.html">
https://scikit-learn.org/stable/modules/feature_selection.html
</a>

<br />
<h4>Libraries</h4>

{% highlight ruby %}

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, mean_squared_error

import tensorflow as tf

%matplotlib inline

{% endhighlight %}

<br />
<h4>Get Data</h4>

{% highlight ruby %}

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())
print('target names ', cancer['target_names'])
# print(cancer['DESCR'])
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
features_for_pca = df.columns.shape[0]
df['target'] = cancer['target']
print(df.groupby('target')['target'].count())

scaler = StandardScaler()
scaler.fit(df.drop('target', axis = 1))
scaled_data = scaler.transform(df.drop('target', axis = 1))

# Replace spaces
column_names  = []
for name in df.columns:
    column_names.append(name.replace(" ", "_")) #_ Added for Atom

df.columns = column_names
# print(df.columns)

df.head()

{% endhighlight %}

<br />
<h4>Logistic Regression L1</h4>

{% highlight ruby %}

X = df.drop('target', axis = 1)
y = df['target']

model = LogisticRegression(penalty='l1')
model.fit(X, y)

coef_dict = {}
for coef, feat in zip(model.coef_[0,:],X):
    if coef != 0: coef_dict[feat] = coef

print(pd.DataFrame.from_dict(coef_dict, orient='index', columns=['Coef']).sort_values(by=['Coef']))

# or

vth = pd.DataFrame({
                    'Name': df.drop('target', axis=1).columns,
                    'VThScore': model.coef_[0]
                   })

print(vth[vth['VThScore'] != 0].sort_values(by=['VThScore']).Name)

{% endhighlight %}

<br />
<h4>VarianceThreshold</h4>

{% highlight ruby %}

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(df.drop('target', axis=1))

print(pd.DataFrame(df[df.columns[sel.get_support(indices=True)]].columns, columns=['Coef']))

{% endhighlight %}

<br />
<h4>SelectKBest</h4>

{% highlight ruby %}

X = df.drop('target', axis = 1)
y = df['target']

selector = SelectKBest(chi2, k=10)
fit_skb = selector.fit(X, y)

skb = pd.DataFrame({
                    'Name': df.drop('target', axis=1).columns,
                    'SKBScore': fit_skb.scores_
                   })

print(skb.sort_values(by='SKBScore', ascending=False).head(10))

# cols = selector.get_support(indices=True)
# print(cols)
# print(df.columns[cols])

{% endhighlight %}

<br />
<h4>Recursive Feature Elimination</h4>

{% highlight ruby %}

model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(X, y)
rfe = pd.DataFrame({
                    'Name': df.drop('target', axis=1).columns,
                    'Rank': fit.ranking_,
                    'Support': fit.support_
                   })

print(rfe.sort_values(by=['Rank']).head(10).sort_index())

{% endhighlight %}

<br />
<h4>Extra Trees</h4>

{% highlight ruby %}

extrees = ExtraTreesClassifier()
extrees.fit(X, y)
extrees = pd.DataFrame({
                    'Name': df.drop('target', axis=1).columns,
                    'ExTrees': extrees.feature_importances_                    
                   })

print(extrees.sort_values(by=['ExTrees']).head(10).sort_index())

{% endhighlight %}

<br />
Hat tip to
<a href="https://www.udemy.com/user/soledad-galli/">
Soledad Galli
</a> onward
<br /><br />
In all feature selection procedures, it is good practice to select the features by examining only the training set. And this is to avoid overfit.

<h4>Misc Methods</h4>

{% highlight ruby %}

# remove constant features
constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0
]

X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)

X_train.shape, X_test.shape

# remove quasi-constant features
sel = VarianceThreshold(
    threshold=0.01)  # 0.1 indicates 99% of observations approximately

sel.fit(X_train)  # fit finds the features with low variance

sum(sel.get_support())

# check for duplicated features in the training set
duplicated_feat = []
for i in range(0, len(X_train.columns)):
    if i % 10 == 0:  # this helps me understand how the loop is going
        print(i)

    col_1 = X_train.columns[i]

    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)

len(duplicated_feat)

# remove duplicated features
X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
X_test.drop(labels=duplicated_feat, axis=1, inplace=True)

X_train.shape, X_test.shape

# find and remove correlated features
# to reduce the feature space

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)

# removed correlated  features
X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)

X_train.shape, X_test.shape

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Step backward greedy selection algorithm

# sfs1 = SFS(RandomForestRegressor(),
#            k_features=10,
#            forward=False,
#            floating=False,
#            verbose=2,
#            scoring='r2',
#            cv=3)

# sfs1 = sfs1.fit(np.array(X_train), y_train)

# Exhaustive feature selector

# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# efs1 = EFS(RandomForestClassifier(n_jobs=4, random_state=0),
#            min_features=1,
#            max_features=4,
#            scoring='roc_auc',
#            print_progress=True,
#            cv=2)

# efs1 = efs1.fit(np.array(X_train[X_train.columns[0:4]].fillna(0)), y_train)

# find important features using univariate roc-auc

# select features using the coefficient of a non
# regularised logistic regression

# from sklearn.feature_selection import SelectFromModel
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html

# sfm = SelectFromModel(LogisticRegression(C=1000))
# sfm.fit(scaler.transform(X, y)

# SelectFromModel(RandomForestClassifier(n_estimators=400))

{% endhighlight %}
