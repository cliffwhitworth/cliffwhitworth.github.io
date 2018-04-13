---
layout: post
title:  "Standard Deviation"
date:   2018-04-03
categories: Stats
---
<br />
<h4>Uncorrected Sample Standard Deviation</h4>

<a href="https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Sample_covariance">
Wikipedia
</a>

{% highlight ruby %}
# from scipy import stats
# from statistics import variance

x_array = [row[0] for row in dataset]
y_array = [row[1] for row in dataset]

print ('Sx: ', np.std(x_array, axis=0))
print ('Sy: ', np.std(y_array, axis=0))
{% endhighlight %}
