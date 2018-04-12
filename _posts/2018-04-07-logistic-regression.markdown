---
layout: post
title:  "Logistic Regression"
date:   2018-04-07
categories: Classification
---
<br />
<h4>Example 1</h4>
<p>Code credit:
<br />
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
</a>
<br />
<a href="https://www.udemy.com/machinelearning/">
https://www.udemy.com/machinelearning/
</a>
</p>

{% highlight ruby %}
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x, y)
{% endhighlight %}
