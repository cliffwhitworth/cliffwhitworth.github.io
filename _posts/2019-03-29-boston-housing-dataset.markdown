---
layout: post
title:  "Boston Housing Dataset"
date:   2019-03-29
categories: More
---
<br />

<a href="http://scikit-learn.org/stable/datasets/index.html">
Dataset Loading Utilities
</a>

<a href="https://nbviewer.jupyter.org/github/cliffwhitworth/machine_learning_notebooks/blob/master/BostonHousingDataset.ipynb">
Notebook
</a>

{% highlight ruby %}
import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm

from sklearn.datasets import load_boston
boston = load_boston()

print('Keys')
print(boston.keys())

# Print the feature names and the description
print('Feature Names')
print(boston.feature_names)
print()
print('Description')
print(boston.DESCR)

# Print the shape of the data and the target
print('Data Shape')
print(boston.data.shape)
print()
print('Target Shape')
print(boston.target.shape)

# Create a dataframe of the data and add the Median Value target
boston_dataframe = pd.DataFrame(boston.data)
boston_dataframe.columns = boston.feature_names
boston_dataframe['MEDV'] = boston.target
print('Dataframe Head')
print(boston_dataframe.head())

# Print descriptives
print(boston_dataframe.describe())

# Ordinary Least Squares Model
ols = sm.OLS(endog = boston.target, exog = boston.data).fit()
ols.summary()


{% endhighlight %}
