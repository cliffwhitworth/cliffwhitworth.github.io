---
layout: post
title:  "Make_Classification for Logistic Regression"
date:   2018-11-02
categories: Classification
---
<br />

<a href="http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html">
Scikit-learn
</a>

<a href="https://chrisalbon.com/machine_learning/basics/make_simulated_data_for_classification/">
Chris Albon
</a>

<a href="https://nbviewer.jupyter.org/github/cliffwhitworth/machine_learning_notebooks/blob/master/LogisticRegression.ipynb">
Notebook
</a>

<h4>Import Libraries</h4>

{% highlight ruby %}

import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

{% endhighlight %}

<h4>Binary Logistic Classification</h4>

{% highlight ruby %}

n = 1000
# n_classes = 2 by default
features, binary_class = make_classification(n_samples=n, n_features=2,  
#                                              weights=[.4, .6], # weights per class
                                             n_informative=1, n_redundant=0, n_clusters_per_class=1)

# Create a dataframe of the features and add the binary class (label, output)
df = pd.DataFrame(features)
df.columns = ['Feature_1', 'Feature_2']
df['Binary_Class'] = binary_class

X = df.drop('Binary_Class', axis=1)
y = df['Binary_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test,predictions))
tn, fp, fn, tp = confusion_matrix(y_test,predictions).ravel()
print(tn, fp, fn, tp)

print()
print('Classification Report')
print(classification_report(y_test,predictions))

print()  # Compare with classification report
print('Accuracy Score')
print(accuracy_score(y_test, predictions))

print() # Compare with classification report
print('F1 Score')
print(f1_score(y_test, predictions, average=None))

{% endhighlight %}

<h4>Multiclass Logistic Regression</h4>

{% highlight ruby %}

n = 1000
# https://chrisalbon.com/machine_learning/basics/make_simulated_data_for_classification/
# Create a simulated feature matrix and output vector with 100 samples
features, multi_class = make_classification(n_samples = n, n_features = 3,
                                       n_informative = 3, # features that actually predict the output's classes
                                       n_redundant = 0, # features that are random and unrelated to the output's classes
#                                        weights = [.2, .3, .5],
                                       n_classes = 3, n_clusters_per_class=1)

# Create a dataframe of the features and add the binary class (label, output)
df = pd.DataFrame(features)
df.columns = ['Feature_1', 'Feature_2', 'Feature_3']
df['Multi_Class'] = multi_class

X = df.drop('Multi_Class', axis=1)
y = df['Multi_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print('Confusion Matrix')
print(confusion_matrix(y_test,predictions))

print()
print('Classification Report')
print(classification_report(y_test,predictions))

print() # Compare with classification report
print('Accuracy Score')
print(accuracy_score(y_test, predictions))

print() # Compare with classification report
print('F1 Score')
print(f1_score(y_test, predictions, average=None))

{% endhighlight %}
