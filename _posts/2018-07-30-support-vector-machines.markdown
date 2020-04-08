---
layout: post
title:  "Support Vector Machines"
date:   2018-07-30 05
categories: Classification
---
<br />

<a href="http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">
SVC
</a>

{% highlight ruby %}

# Split
# Scale

# Fit
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

{% endhighlight %}
