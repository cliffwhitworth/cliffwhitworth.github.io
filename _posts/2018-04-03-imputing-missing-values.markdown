---
layout: post
title:  "Imputing Missing Values"
date:   2018-04-03 02
categories: More
---
<br />

<a href="https://nbviewer.jupyter.org/github/cliffwhitworth/machine_learning_notebooks/blob/master/ImputingMissingValues.ipynb">
Notebook
</a>

<h4>Mean</h4>

{% highlight ruby %}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataframe
df = pd.DataFrame()

n = 100

# Make individuals
df['Individual'] = np.random.randint(low=20, high=60, size=n)

# Delete some individual values
df = df.mask(np.random.random(df.shape) < .1)

# Make groups
df['Group'] = np.random.randint(low=1, high=4, size=n)

# Visualize missing data with heatmap (hat tip to Jose Portilla)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='plasma')

# Provide average based on group of missing numbers
def get_feature_mean(row, feature, index):
    if row.index[index] == feature:
        if np.isnan(row[row.index[index]]):
            return df.loc[df['Group'] == row.Group, feature].mean().round(0)
        else:
            return row[row.index[index]]

feature = 'Individual'
df[feature] = df.apply(get_feature_mean, args=(feature, df.columns.get_loc(feature)), axis=1)

{% endhighlight %}
