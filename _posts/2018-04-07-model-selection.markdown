---
layout: post
title:  "Model Selection"
date:   2018-04-07 10
categories: More
---
<br />
<h4>Grid Search</h4>
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV">
Grid Search
</a>
<br />
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">
SVC Parameters Example
</a>

{% highlight ruby %}

# using grid search for sklearn.svm.SVC
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

{% endhighlight %}

<br />
<h4>K-Fold Cross Validation</h4>
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score">
Sklearn
</a>
<br />
<a href="https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6">
Towards Data Science
</a>
<br />
<a href="https://www.kaggle.com/dansbecker/cross-validation">
Kaggle
</a>

{% highlight ruby %}

# imports needed
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

scores = cross_val_score(estimator, X, y, cv=10)
print ("Cross-validated scores:", scores)
r2 = metrics.r2_score(y, yhat)
print ("R-squared:", r2)

{% endhighlight %}

<br />
<h4>Test Train Split</h4>
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">
Test Train Split
</a>
<br />
<a href="https://www.udemy.com/machinelearning/">
https://www.udemy.com/machinelearning/
</a>

{% highlight ruby %}

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

{% endhighlight %}
