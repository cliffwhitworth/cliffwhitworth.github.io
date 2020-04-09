---
layout: post
title: "Adabooster Classifier"
date: 2020-04-09 16:25:06
comments: false
---

* [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [Usage](https://nbviewer.jupyter.org/github/cliffwhitworth/machine_learning_notebooks/blob/master/CensusPredictions.ipynb)

Adaboost Classifier

```
# Optimize AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# Provide hyperparameters
h_params = {'n_estimators':[50,120],
           'learning_rate':[0.1,0.5,1.],
           'base_estimator__min_samples_split':np.arange(2,8,2),
           'base_estimator__max_depth':np.arange(1,4,1)}
```
base_estimator uses DecisionTreeClassifier by default with min_samples_split default set to 2. See [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

```
scorer = make_scorer(fbeta_score,beta=0.5)
grid_obj = GridSearchCV(clf, h_params, scorer)
grid_fit = grid_obj.fit(X_train, y_train)
best_clf = grid_fit.best_estimator_
```
make_scorer wraps scoring functions for GridSearchCV. See [make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html). fbeta_score is the weighted harmonic mean of precision and recall. The beta parameter controls the weighting. See [fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)
```
predictions = best_clf.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, predictions):.4f}')
print(f'F-score: {fbeta_score(y_test, predictions, 0.5):.4f}')
print('\nBest Model\n-----')
print(best_clf)
```



