---
layout: post
title:  "Dimensionality Reduction"
date:   2019-04-07 05
categories: More
---
<br />
<h4>Kernel PCA</h4>

<a href="http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html">
Sklearn
</a>
<br />
<a href="https://www.kaggle.com/lambdaofgod/kernel-pca-examples">
Kaggle Examples
</a>

{% highlight ruby %}

  # Split the dataset
  # Feature scaling

  # Apply Kernel
  from sklearn.decomposition import KernelPCA
  # kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
  # X_kpca = kpca.fit_transform(X)
  # X_back = kpca.inverse_transform(X_kpca)
  # pca = PCA()
  # X_pca = pca.fit_transform(X)
  kpca = KernelPCA(n_components = 2, kernel = 'rbf')
  X_train = kpca.fit_transform(X_train)
  X_test = kpca.transform(X_test)

  # Fit Logistic Regression to the Training set

{% endhighlight %}

<br />
<h4>Linear Discriminant Analysis</h4>

<a href="http://scikit-learn.org/stable/modules/lda_qda.html">
Sklearn
</a>
<br />
<a href="https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/">
Linear Discriminant Analysis
</a>
<br />
<a href="https://elitedatascience.com/dimensionality-reduction-algorithms">
Supervised Feature Extraction
</a>

{% highlight ruby %}

  # Split the dataset
  # Feature scaling

  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

  lda = LDA(n_components = 2)
  # X_r2 = lda.fit(X, y).transform(X)
  X_train = lda.fit_transform(X_train, y_train)
  X_test = lda.transform(X_test)

  # Fit Logistic Regression to the Training set

{% endhighlight %}

<br />
<h4>Principal Component Analysis</h4>

<a href="http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py">
Sklearn
</a>
<br />
<a href="http://setosa.io/ev/principal-component-analysis/">
Principal Component Analysis
</a>
<br />
<a href="https://elitedatascience.com/dimensionality-reduction-algorithms">
Unsupervised Feature Extraction
</a>

{% highlight ruby %}

  # Split the dataset
  # Feature scaling

  from sklearn.decomposition import PCA

  pca = PCA(n_components=2)
  # X_r = pca.fit(X).transform(X)
  X = pca.fit_transform(X_train)
  X = pca.transform(X_test)
  explained_variance = pca.explained_variance_ratio_

  # Percentage of variance explained for each components
  print('explained variance ratio (first two components): %s'
        % str(pca.explained_variance_ratio_))

  # Fit Logistic Regression to the Training set

{% endhighlight %}

<br />
<h4>Explained Variance Example</h4>

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

{% endhighlight %}
