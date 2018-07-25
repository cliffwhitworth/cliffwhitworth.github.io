---
layout: post
title:  "Agglomerative Hierarchical Clustering"
date:   2018-05-24 04
categories: Clustering
---
<br />
<h4>Agglomerative Hierarchical Clustering</h4>
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html">
Sklearn
</a>
<br />
<a href="http://scikit-learn.org/stable/modules/clustering.html">
Overview of Clustering
</a>
<br />
<a href="https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68">
5 Clustering Algorithms Data Scientists Need to Know
</a>


{% highlight ruby %}

# Fit
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = n, affinity = 'euclidean', linkage = 'ward')
y = ac.fit_predict(X)


{% endhighlight %}
