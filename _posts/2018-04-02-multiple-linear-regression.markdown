---
layout: post
title:  "Multiple Linear Regression"
date:   2018-04-02
categories: Regression
---
<br />
<h4>Example 1</h4>
<p>Code credit:
<br />
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
</a>
<br />
<a href="https://www.udemy.com/machinelearning/">
https://www.udemy.com/machinelearning/
</a>
</p>

{% highlight ruby %}
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
{% endhighlight %}
