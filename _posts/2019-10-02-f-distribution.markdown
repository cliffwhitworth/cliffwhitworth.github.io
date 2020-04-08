---
layout: post
title:  "F-Distribution"
date:   2019-10-02 04
categories: Stats
---
<br />
<h4>F Score</h4>

{% highlight ruby %}

from scipy import stats
stats.f.ppf(1-.05,dfn=2,dfd=27)

# Excel
# FINV(alpha,df1,df2)

{% endhighlight %}
