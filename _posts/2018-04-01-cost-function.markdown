---
layout: post
title:  "Cost Function"
date:   2018-04-01
categories: More
---
<br />
<h4>Cost Function</h4>

<a href="http://ozzieliu.com/2016/02/09/gradient-descent-tutorial/">
Ozzie Liu's Example
</a>

{% highlight ruby %}

def cost_function(X, y, theta):

    m = y.size
    # 1/2m * sum of (predictions - y)^2
    J = ((X.dot(theta) - np.vstack(y.T)) ** 2).sum()/(2 * m)
    return J

{% endhighlight %}
