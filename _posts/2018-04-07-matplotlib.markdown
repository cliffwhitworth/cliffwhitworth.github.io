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
