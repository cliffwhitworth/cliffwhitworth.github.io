---
layout: post
title:  "Random Forest Regression"
date:   2018-05-03
categories: Regression
---
<br />
<h4>Random Forest Regression</h4>
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">
Sklearn
</a>

{% highlight ruby %}

# No Split
# No Scale

# Fit
from sklearn.ensemble import RandomForestRegressor
# regr = RandomForestRegressor(max_depth=2, random_state=0)
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

{% endhighlight %}
