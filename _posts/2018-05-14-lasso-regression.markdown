---
layout: post
title:  "Lasso Regression"
date:   2018-05-14 01
categories: Regression
---
<br />

<a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html">
Sklearn
</a>
<br />
<a href="https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/">
A Comprehensive Beginner's Guide
</a>

{% highlight ruby %}

# Import
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import linear_model

# Fit
regressor = linear_model.LassoCV(cv=5,normalize=True,random_state=10,alphas=[.0005])
regressor.fit(X_train, y_train)

# Metrics
yhat = regressor.predict(X_test)
print(metrics.mean_squared_error(y_true=y_test, y_pred=yhat))
print(r2_score(y_test, yhat))

{% endhighlight %}
