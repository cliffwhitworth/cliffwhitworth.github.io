---
layout: post
title:  "Correlation Coefficient"
date:   2018-04-01
categories: Stats
---
<br />
<h4>Sample Correlation Coefficient</h4>

<a href="https://en.wikipedia.org/wiki/Correlation_and_dependence#Pearson's_product-moment_coefficient">
Wikipedia
</a>
</p>

{% highlight ruby %}
# from scipy import stats
# from statistics import variance

x_array = [row[0] for row in dataset]
y_array = [row[1] for row in dataset]


print ('Stats Pearsonr: ', stats.pearsonr(x_array, y_array)[0])
print ('Numpy CorrCoef: ', np.corrcoef(x_array, y_array)[0,1])

{% endhighlight %}
