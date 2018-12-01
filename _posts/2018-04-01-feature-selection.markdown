---
layout: post
title:  "Feature Selection"
date:   2018-04-01 09
categories: More
---
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
from sklearn.feature_selection import RFE, SelectKBest, f_regression
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
    column_names.append(name.replace(" ", "_"))

df.columns = column_names
# print(df.columns)

df.head()

{% endhighlight %}

<br />
<h4>SelectKBest</h4>

{% highlight ruby %}

{% endhighlight %}

<br />
<h4>Recursive Feature Elimination</h4>

{% highlight ruby %}

{% endhighlight %}

<br />
<h4>Extra Trees</h4>

{% highlight ruby %}

{% endhighlight %}

<br />
Hat tip to
<a href="https://www.udemy.com/user/soledad-galli/">
Soledad Galli
</a> onward

<br />
<h4>Filter Methods</h4>

{% highlight ruby %}

{% endhighlight %}

<br />
<h4>Wrapper Methods</h4>

{% highlight ruby %}

{% endhighlight %}

<br />
<h4>Embedded Methods</h4>

{% highlight ruby %}

{% endhighlight %}
