---
layout: post
title:  "Decision Tree Classification"
date:   2018-04-28
categories: Classification
---
<br />
<h4>Decision Tree Classification</h4>
<a href="http://scikit-learn.org/stable/modules/tree.html#classification">
Decision Tree Classification
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

# fit the data
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# how are the predictions looking on the test set
yhat = classifier.predict(X_test)

{% endhighlight %}
