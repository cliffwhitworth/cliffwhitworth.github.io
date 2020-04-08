---
layout: post
title:  "Student T-Test"
date:   2018-05-17 04
categories: Stats
---
<br />
<h4>One Tail Two Tail</h4>

{% highlight ruby %}

from scipy.stats import ttest_ind

ttest_ind(a,b).statistic

# Two tail
ttest_ind(a,b).pvalue/2

{% endhighlight %}
