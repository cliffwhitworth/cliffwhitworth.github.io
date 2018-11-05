---
layout: post
title:  "Logistic Regression"
date:   2018-04-07
categories: Classification
---
<br />

<a href="https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html">
Read the Docs
</a>
<br />
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">
Sklearn
</a>
<br />
<a href="https://en.wikipedia.org/wiki/Logistic_regression">
Wikipedia
</a>
<br />
<a href="https://github.com/cliffwhitworth/machine_learning_notebooks/blob/master/LogisticRegressionNotes.ipynb">
Notebook
</a>

{% highlight ruby %}

# Libraries

import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

# classifier = LogisticRegression()
# classifier.fit(x, y)

{% endhighlight %}
<br />

<h4>Notes</h4>

<ul>
<li>Make classification:
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html">
Make classification
</a>
</li>
<li>Equation:
<a href="http://www.saedsayad.com/logistic_regression.htm">
Dr. Saed Sayad
</a>
</li>
<li>Log odds:
<a href="https://www.statisticshowto.datasciencecentral.com/log-odds/">
Statistics how to
</a>
</li>
<li>MLE:
<a href="https://www.bogotobogo.com/python/scikit-learn/Maximum-Likelyhood-Estimation-MLE.php">
Bogotobogo
</a>
</li>
<li>Sigmoid function:
<a href="https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6">
Towards data science
</a>
</li>
<li>The S curve:
<a href="https://www.goodreads.com/author/quotes/3242685.Pedro_Domingos?page=5">
Bogotobogo
</a>
</li>
</ul>
<br />

<h4>Generate Data</h4>

{% highlight ruby %}

# Generate a random n-class classification problem
n = 1000
X, y = make_classification(n_samples=n, n_features=1,  
                                             n_informative=1, n_redundant=0, n_clusters_per_class=1)

# Create a dataframe of the feature and class
df = pd.DataFrame({'Feature': X.flatten(), 'Class': y.flatten()})
print('Dataframe Head')
print(df.head())
print()

{% endhighlight %}
<br />

<h4>Model Summary with Statsmodels</h4>

{% highlight ruby %}

print('Model Summary')
import statsmodels.api as sm
logit=sm.Logit(df['Class'], df['Feature'])
result=logit.fit()
print(result.summary2())

{% endhighlight %}
<br />

<h4>Model with Sklearn</h4>

{% highlight ruby %}

nX = df['Feature'].values.reshape(-1,1)
ny = df['Class'].values
model = LogisticRegression(C=1e9, solver='lbfgs')
model.fit(nX, ny)
predictions = model.predict(nX)

print('Iterations: ', model.n_iter_)
print('Intercept: ', model.intercept_)
print('Coefficient: ', model.coef_)
print()

print('Accuracy Score')
print(accuracy_score(ny, predictions))
print()

print('Confustion matrix')
print(confusion_matrix(ny,predictions))
tn, fp, fn, tp = confusion_matrix(ny,predictions).ravel()
print('tn: {} fp: {} fn: {} tp: {}'.format(tn, fp, fn, tp))
print()

print('Classification Report')
print(classification_report(ny,predictions))
print()

print('Misclassified count and location')
misclassified = np.flatnonzero(ny != predictions)
print(len(misclassified), misclassified)
print()
print(df.loc[misclassified,:])

{% endhighlight %}
<br />

<h4>Sklearn / Statsmodels Comparison</h4>

{% highlight ruby %}

# https://stats.stackexchange.com/questions/203740/logistic-regression-scikit-learn-vs-statsmodels
# sklearn
mdl = LogisticRegression(C=1e9, solver='lbfgs')
sklearn = mdl.fit(nX, ny)
print('Sklearn coef_: ', sklearn.coef_.flatten()[0])
print()

# sm
statsmodel = sm.Logit(df['Class'], df['Feature'])
print(statsmodel.fit().params)

{% endhighlight %}
<br />

<h4>ROC / AUC</h4>

<a href="https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8">Towards data science</a>

{% highlight ruby %}

# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
logit_roc_auc = roc_auc_score(ny, model.predict(nX))
fpr, tpr, thresholds = roc_curve(ny, model.predict_proba(nX)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

{% endhighlight %}
<br />

<h4>Logistic / Linear Regression Plot</h4>

<a href="http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py">Logistic / linear plog</a>

{% highlight ruby %}

# Code source: Gael Varoquaux
# License: BSD 3 clause

# Plot the logistic and linear models
plt.figure(figsize=(12, 4))
plt.clf()
plt.scatter(df['Feature'], df['Class'], color='black', s=2)
X_line = np.linspace(-3, 3, 100)

def model_func(x):
    return 1 / (1 + np.exp(-x))

loss = model_func(X_line * model.coef_ + model.intercept_).ravel()
plt.plot(X_line, loss, color='red', linewidth=1)

ols = LinearRegression()
ols.fit(nX, df['Class'])
plt.plot(X_line, ols.coef_ * X_line + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')

plt.ylabel('Class')
plt.xlabel('Feature')
plt.xticks(range(-3, 3))
plt.yticks([0, 0.5, 1])
plt.legend(('Logistic Regression Model', 'Linear Regression Model'), loc="lower right")
plt.tight_layout()
plt.show()

{% endhighlight %}
