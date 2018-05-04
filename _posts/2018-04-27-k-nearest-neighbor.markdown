---
layout: post
title:  "K-Nearest Neighbor"
date:   2018-04-27 01
categories: Classification
---
<br />
<h4>K-Nearest Neighbor</h4>
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier">
Sklearn
</a>

{% highlight ruby %}

# Split
# Scale

# Fit
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

{% endhighlight %}
