---
layout: post
title:  "Elastic Net Regression"
date:   2018-09-14 22
categories: Regression
---
<br />

<a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html">
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
regressor = linear_model.ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
regressor.fit(X_train, y_train)

# Metrics
yhat = regressor.predict(X_test)
print(metrics.mean_squared_error(y_true=y_test, y_pred=yhat))
print(r2_score(y_test, yhat))

{% endhighlight %}
