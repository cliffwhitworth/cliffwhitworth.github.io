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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

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
<h4>Find Explained Variance</h4>

{% highlight ruby %}

# https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/

pca_full = PCA(n_components=features_for_pca)
pca_full.fit(scaled_data)

# Amount of variance for each component
var = pca_full.explained_variance_ratio_

# Cumulative sum
cumsum = np.cumsum(np.round(pca_full.explained_variance_ratio_, decimals=4)*100)

# Find the number of pca components that account for 95% (~10)
plt.figure(figsize=(12, 4))

# plt.subplot(nrows=1, ncols=3, nplt=1)
plt.subplot(121)
plt.plot(var, 'b-')
plt.title('Variance')
plt.xlabel('Components')
plt.ylabel('% Explained Variance')
plt.grid()

plt.subplot(122)
plt.plot(cumsum, 'r-')
plt.title('Cumulative Sum')
plt.xlabel('Components')
plt.ylabel('Sum % Explained Variance')
plt.grid()

plt.show()

{% endhighlight %}

<br />
<h4>Principal Component Analysis</h4>

{% highlight ruby %}

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
