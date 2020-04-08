---
layout: post
title:  "Chi Square"
date:   2019-01-17 03
categories: Stats
---
<br />
<h4>Critical Value</h4>

{% highlight ruby %}

from scipy.stats import chi2
chi2.isf(0.5,5)

# Excel
# 95% confidence and 5 degrees of freedom
# CHISQ.INV.RT(0.05,5)

{% endhighlight %}
