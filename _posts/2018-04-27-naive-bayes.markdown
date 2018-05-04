---
layout: post
title:  "Naive Bayes"
date:   2018-04-27 15
categories: Classification
---
<br />
<h4>Naive Bayes</h4>
<a href="http://scikit-learn.org/stable/modules/naive_bayes.html">
Sklearn
</a>

{% highlight ruby %}

# Split
# Scale

# Fit
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

{% endhighlight %}
