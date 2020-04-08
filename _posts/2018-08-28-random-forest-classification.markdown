---
layout: post
title:  "Random Forest Classification"
date:   2018-08-28 05
categories: Classification
---
<br />

<a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier">
Sklearn
</a>
<br />
<h4>Entropy</h4>
<a href="https://bricaud.github.io/personal-blog/entropy-in-decision-trees/">
Entropy
</a>
<br />
<a href="https://nullpointerexception1.wordpress.com/2017/12/13/entropy-in-machine-learning/">
Entropy
</a>

{% highlight ruby %}

# Split
# Scale

# Fit
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

{% endhighlight %}
