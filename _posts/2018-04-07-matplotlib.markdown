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
