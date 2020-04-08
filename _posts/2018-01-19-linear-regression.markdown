---
layout: post
title:  "Linear Regression"
date:   2018-01-19 05
categories: Stats
---
<br />
<h4>Slope and Intercept</h4>

{% highlight ruby %}

from scipy.stats import linregress
slope = round(lingress(x,y).slope,1)
intercept = round(lingress(x,y).intercept,1)

print(f'y = {intercept} + {slope}x')

# Excel Data Analysis Analysis Tools
# Regression

# Multiple Regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(list(zip(x1,x2)),y)
b1,b2 = reg.coef_[0], reg.coef_[1]
b0 = reg.intercept_

print(f'y = {b0:.{3}} + {b1:.{3}}x1 + {b2:.{3}}x2')

{% endhighlight %}
