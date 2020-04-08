---
layout: post
title:  "Distributions"
date:   2019-08-02 04
categories: Stats
---
<br />
<h4>Binomial Distribution</h4>

{% highlight ruby %}

from scipy.stats import binom
binom.pmf(x,n,p) # probability mass function

# Excel
# If you roll a die 16 times, what is the probability that a five comes up 3 times?
# x=3(success);n=16;p=1/6
# BINOM.DIST(x,n,p,FALSE)

{% endhighlight %}

<br />
<h4>Poisson Distribution</h4>

{% highlight ruby %}

from scipy.stats import poisson
poisson.pmf(n,x)

# Fewer than 3
poisson.cdf(2,8) # cumulative distribution function

# No deliveries between 4 and 4:05...
poisson.pmf(0,8/12)

# Excel
# What is the probability that only 4 deliveries will arrive between 4 and 5pm this Friday?
# POISSON.DIST(x,n,FALSE)

# What is the probability that fewer than 3 will arrive...
# POISSON.DIST(x,n,TRUE)

{% endhighlight %}

<br />
<h4>Normal Distribution</h4>

{% highlight ruby %}

from scipy import stats
stats.norm.cdf(z)
stats.norm.ppf(p)

# Excel
# input z NORMDIST(z) output p
# input p NORMSINV(p) output z

{% endhighlight %}
