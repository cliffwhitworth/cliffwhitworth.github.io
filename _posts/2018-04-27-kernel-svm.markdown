---
layout: post
title:  "Kernel SVM"
date:   2018-04-27 10
categories: Classification
---
<br />
<h4>Kernel SVM</h4>
<a href="http://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#sphx-glr-auto-examples-svm-plot-custom-kernel-py">
Sklearn
</a>

{% highlight ruby %}

# Split
# Scale

# Fit
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

{% endhighlight %}
