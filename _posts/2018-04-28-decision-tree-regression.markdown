---
layout: post
title:  "Decision Tree Regression"
date:   2018-04-28
categories: Regression
---
<br />
<h4>Support Vector Regression</h4>
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
