---
layout: post
title:  "Standard Deviation"
date:   2018-04-03 02
categories: Stats
---
<br />
<h4>Uncorrected Sample Standard Deviation</h4>

<a href="https://en.m.wikipedia.org/wiki/Standard_deviation">
Wikipedia
</a>

{% highlight ruby %}

x_array = [row[0] for row in dataset]
y_array = [row[1] for row in dataset]

print ('Sx: ', np.std(x_array, axis=0))
print ('Sy: ', np.std(y_array, axis=0))
{% endhighlight %}
