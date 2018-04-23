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

    # 1/2m * sum of (predictions - y)^2
    J = ((X.dot(theta) - np.vstack(y.T)) ** 2).sum()/(2 * y.size)
    return J

{% endhighlight %}

<br />
<h4>Cost Function Example</h4>

<a href="https://scipython.com/blog/visualizing-the-gradient-descent-method/">
Code that got this started
</a>


{% highlight ruby %}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as plb

def cost_func(theta):
    theta = np.atleast_2d(np.asarray(theta))
    return np.average((y-(x*theta))**2, axis=1)/2

# dataset
dataset = [[1, 1], [2, 3], [3, 2], [4, 3], [5, 5]]

# x and y arrays
x = np.array([row[0] for row in dataset])
y = np.array([row[1] for row in dataset])

# Take N steps with learning rate alpha
# down the steepest gradient, starting at theta = 0 with m size.
N = 15
alpha = .1
m = 10
theta=[0]
J = [cost_func(theta[0])[0]]

for j in range(N-1):
    last_theta = theta[-1]
    this_theta = last_theta - alpha / m * np.sum(((x*last_theta) - y) * x)
    theta.append(this_theta)
    J.append(cost_func(this_theta))

# print cost and theta
print('Initial cost: ', J[0])
print('Cost min: ', J[-1][0])
print('Theta: ', theta[-1])

{% endhighlight %}
