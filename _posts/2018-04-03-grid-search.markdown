---
layout: post
title:  "Grid Search"
date:   2018-04-03 01
categories: More
---
<br />

<a href="https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search">
Grid Search
</a>

{% highlight ruby %}

# Entropy
import time
parameters = {'max_depth': [3, None],
              'max_features': [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              'bootstrap': [True, False],
              'criterion': ['entropy']}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print('Took %0.2f seconds' % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters

print('best accuracy', rf_best_accuracy)
print('best parameters', rf_best_parameters)

{% endhighlight %}

{% highlight ruby %}

# Gini
parameters = {'max_depth': [3, None],
              'max_features': [1, 5, 10],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 5, 10],
              'bootstrap': [True, False],
              'criterion': ['gini']}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print('Took %0.2f seconds' % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters

print('best accuracy', rf_best_accuracy)
print('best parameters', rf_best_parameters)

{% endhighlight %}