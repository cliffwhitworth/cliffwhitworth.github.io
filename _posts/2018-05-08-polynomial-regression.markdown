---
layout: post
title:  "Polynomial Regression"
date:   2018-05-08
categories: Regression
---
<br />
<h4>Example 1</h4>
<p>Code credit:
<br />
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">
Scikit-learn
</a>
<br />
<a href="https://www.udemy.com/machinelearning/">
https://www.udemy.com/machinelearning/
</a>
</p>

{% highlight ruby %}

# No Split
# No Scale

# Fit
from sklearn.preprocessing import PolynomialFeatures
polynom_reg = PolynomialFeatures(3) # degree = 3
polynom_reg.fit_transform(x)

{% endhighlight %}
