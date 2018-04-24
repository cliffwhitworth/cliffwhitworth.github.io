---
layout: post
title:  "MatPlotLib"
date:   2018-04-07 02
categories: More
---
<br />
<a href="https://matplotlib.org/">
https://matplotlib.org/
</a>
<br />
<h4>Scatterplot with Line of Best Fit</h4>

{% highlight ruby %}

plt.scatter(x, y)
plt.plot(x, yhat)
# plt.axis([0, 6, 0, 6])
plt.grid(True)
plt.show()

{% endhighlight %}

<br />
<h4>3D Plot</h4>

{% highlight ruby %}

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], Y)
plt.show()

{% endhighlight %}

<br />
<h4>Subplots</h4>
<a href="https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html">
Subplots
</a>
<br />
{% highlight ruby %}

plt.figure(1)

plt.subplot(131)
plt.plot(features[:, 0], Y, 'bo')
plt.title('R&D')

plt.subplot(132)
plt.plot(features[:, 1], Y, 'ro')
plt.title('Admin')

plt.subplot(133)
plt.plot(features[:, 2], Y, 'yo')
plt.title('Marketing')

plt.show()

{% endhighlight %}

<br />
<h4>Contour Map, Probability Grid, Decision Boundary</h4>
<a href="https://stackoverflow.com/questions/20045994/how-do-i-plot-the-decision-boundary-of-a-regression-using-matplotlib">
Stackoverflow
</a>
<br />
<a href="https://stackoverflow.com/questions/28256058/plotting-decision-boundary-of-logistic-regression">
Stackoverflow
</a>
<br />
<a href="https://www.kunxi.org/notes/machine_learning/logistic_regression/">
www.kunxi.org
</a>
<br />
{% highlight ruby %}

X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = 0.01))

#plot background colors
ax = plt.gca()
Z = classifier.predict_proba(np.c_[X1.ravel(), X2.ravel()])[:, 1]
Z = Z.reshape(X1.shape)
cs = ax.contourf(X1, X2, Z, cmap='RdBu', alpha=.2)
cs2 = ax.contour(X1, X2, Z, cmap='RdBu', alpha=.6)
plt.clabel(cs2, colors = 'k', fontsize=14)

# Plot the points
ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='Class 1')
ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label='Class 2')

# make legend
plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
plt.show()

{% endhighlight %}

<br />
<h4>Bowls, Labels, Arrows, Major / Minor Ticks</h4>
<br />

{% highlight ruby %}

from matplotlib import colors as mcolors

# Start the plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,5))

# Setup convex shape
theta_grid = np.linspace(-0.5,2.5,50)
J_grid = cost_func(theta_grid[:,np.newaxis])

# The cost function as a function of its single parameter, theta.
ax.plot(theta_grid, J_grid, 'k')

# cost function plot with coloured points indicating the
# parameters chosen and red arrows indicating the steps down the gradient.
colors = ['b', 'g', 'm', 'c', 'orange']

for j in range(1,N):
    ax.annotate('', xy=(theta[j], J[j]), xytext=(theta[j-1], J[j-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

# Labels, titles and a legend.
ax.scatter(theta, J, c=colors)
ax.set_xlim(-0.5,2.5)
ax.set_ylim(0, 6)
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$J(\theta_1)$')
ax.set_title('Cost Function')

# major ticks every 1, minor ticks every .1
major_yticks = np.arange(0, 6, 1)
minor_yticks = np.arange(0, 6, .1)
major_xticks = np.arange(-0.5, 2.5, 1)
minor_xticks = np.arange(-0.5, 2.5, .1)

ax.set_yticks(major_yticks)
ax.set_yticks(minor_yticks, minor=True)
ax.set_xticks(major_xticks)
ax.set_xticks(minor_xticks, minor=True)

# different settings for the grids:
ax.grid(which='minor', alpha=0.4)
ax.grid(which='major', alpha=1)

plt.xticks(np.arange(-.5, 2.5, .5))
plt.tight_layout()
plt.grid(True)
plt.show()

{% endhighlight %}

<br />
<h4>Contour Pyplot</h4>

<a href="https://scipython.com/blog/visualizing-the-gradient-descent-method/">
Code that got this started
</a>
<br />
<a href="https://matplotlib.org/api/pyplot_api.html">
Pyplot API
</a>
<br />
{% highlight ruby %}

import pylab as plb
from matplotlib import colors as mcolors

# compute gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        theta = theta - alpha*(1.0/y.size) * np.transpose(X).dot(X.dot(theta) - np.transpose([y]))
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history

# compute cost for linear regression
def compute_cost(X, y, theta):
    J = 0
    s = np.power((X.dot(theta) - np.transpose([y])), 2)
    J = (1.0 / (2  *y.size)) * s.sum(axis = 0)
    return J

# add the bias column to X
X = np.ones(shape=(y.size, 2))
X[:, 1] = x

# theta parameters
thetaP = np.zeros(shape=(2, 1))

# compute and display initial cost
cost = cost_function(X, y, thetaP)

# gradient descent
thetaP, J_history = gradient_descent(X, y, thetaP, alpha, iterations)

# grid over which we will calculate J
theta0 = np.linspace(-2, 2, 100)
theta1 = np.linspace(-2, 2, 100)

# initialize J of theta to a matrix of 0's
Jtheta = np.zeros(shape=(theta0.size, theta1.size))

# fill out J of theta
for t1, element in enumerate(theta0):
    for t2, element2 in enumerate(theta1):
        thetaT = np.zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        Jtheta[t1, t2] = compute_cost(X, y, thetaT)

# tranpose J of theta
Jtheta = Jtheta.T

# contour plotting
plb.contour(theta0, theta1, Jtheta, np.logspace(-2, 3, 20))

for j in range(1,N):
    plb.annotate('', xy=theta[j], xytext=theta[j-1],
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

plb.scatter(*zip(*theta), c=colors)

plb.minorticks_on()
plb.tick_params(axis='x', which='both')

plb.xlabel(r'$\theta_0$')
plb.ylabel(r'$\theta_1$')
plb.scatter(thetaP[0][0], thetaP[1][0])
plb.grid(True, which='both')
plb.show()

{% endhighlight %}

<br />
<h4>A Theta Fit</h4>

<a href="https://scipython.com/blog/visualizing-the-gradient-descent-method/">
Code that got this started
</a>
<br />
{% highlight ruby %}

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# dataset
dataset = [[1, 1], [2, 3], [3, 2], [4, 3], [5, 5]]

# x and y arrays
x = np.array([row[0] for row in dataset])
y = np.array([row[1] for row in dataset])

# The plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
ax.scatter(x, y, marker='x', s=40, color='k')

def cost_function(theta1):
    theta1 = np.atleast_2d(np.asarray(theta1))
    return np.average((y-(x*theta1))**2, axis=1)/2

# Take N steps with learning rate alpha down the steepest gradient, starting at theta1 = 0.
N = 5
alpha = .1
theta = [0]
m = 10

J = [cost_function(theta[0])[0]]
for j in range(N-1):
    last_theta = theta[-1]
    this_theta = last_theta - alpha / m * np.sum(((x*last_theta) - y) * x)
    theta.append(this_theta)
    J.append(cost_function(this_theta))

# Get some colors
colors = ['b', 'g', 'm', 'c', 'orange']
all_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

for value in all_colors:
    colors.append(value)

ax.plot(x, (x*theta[0]), color=colors[0], lw=2, label=r'$\theta_1 = {:.3f}$'.format(theta[0]))

for j in range(1,N):
    ax.plot(x, (x*theta[j]), color=colors[j], lw=2, label=r'$\theta_1 = {:.3f}$'.format(theta[j]))

# Labels, titles and a legend.
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('A Theta Fit')
ax.legend(loc='upper left', fontsize='small')

plt.tight_layout()
plt.show()

{% endhighlight %}
