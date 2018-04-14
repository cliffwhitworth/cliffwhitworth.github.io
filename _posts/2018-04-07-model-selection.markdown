---
layout: post
title:  "Model Selection"
date:   2018-04-07
categories: More
---
<br />
<h4>Test Train Split</h4>
<p>Code credit:
<br />
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">
Test Train Split
</a>
<br />
<a href="https://www.udemy.com/machinelearning/">
https://www.udemy.com/machinelearning/
</a>
</p>

{% highlight ruby %}

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

{% endhighlight %}
