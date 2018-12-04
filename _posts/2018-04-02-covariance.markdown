---
layout: post
title:  "Covariance"
date:   2018-04-02 02
categories: Stats
---
<br />
<h4>Sample Covariance</h4>

<a href="https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Sample_covariance">
Wikipedia
</a>

{% highlight ruby %}

x_array = [row[0] for row in dataset]
y_array = [row[1] for row in dataset]

Join a sequence of arrays along a new axis.
xy_stacked = np.stack((x_array, y_array))

print('Covariance: ', np.cov(xy_stacked)[0][1])

{% endhighlight %}
