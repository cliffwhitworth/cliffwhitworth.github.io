---
layout: post
title:  "Feature Scaling"
date:   2018-04-02 10
categories: More
---
<br />
<a href="http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py">
List of Scalars
</a>
<br />
<a href="http://benalexkeen.com/feature-scaling-with-scikit-learn/">
Some formulas
</a>
<br />
<h4>Standard Scalar</h4>
<p>Code credit:
<br />
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">
Scikit-learn
</a>
<br />
<a href="https://www.udemy.com/machinelearning/">
https://www.udemy.com/machinelearning/
</a>
</p>

{% highlight ruby %}

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

{% endhighlight %}
