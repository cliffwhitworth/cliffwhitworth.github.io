---
layout: post
title:  "Support Vector Regression"
date:   2018-08-02 01
categories: Regression
---
<br />

<a href="http://scikit-learn.org/stable/modules/svm.html#regression">
Support Vector Regression
</a>
<br />
<a href="http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py">
Examples
</a>

{% highlight ruby %}

# No Split
# Scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))

# Fit
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# predictions
yhat = regressor.predict(6.5)
yhat = sc_y.inverse_transform(yhat)

{% endhighlight %}
