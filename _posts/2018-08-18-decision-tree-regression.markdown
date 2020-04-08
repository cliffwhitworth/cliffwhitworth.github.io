---
layout: post
title:  "Decision Tree Regression"
date:   2018-08-18 01
categories: Regression
---
<br />

<a href="http://scikit-learn.org/stable/modules/tree.html#regression">
Sklearn
</a>

{% highlight ruby %}

# No Split
# No Scale

# Fit
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# make a prediction
yhat = regressor.predict(?)

{% endhighlight %}
