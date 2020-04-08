---
layout: post
title:  "Cross Validation"
date:   2019-05-23 04
categories: More
---
<br />

<a href="https://scikit-learn.org/stable/modules/cross_validation.html">
Cross Validation
</a>

{% highlight ruby %}

# Use Random Forest 
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'entropy')
model.fit(X_eda, y_eda[0])

# K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X= X_eda, y = y_eda[0], cv = 10)

print('Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)'  % (accuracies.mean(), accuracies.std() * 2))

{% endhighlight %}