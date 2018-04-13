---
layout: post
title:  "Simple Linear Regression"
date:   2018-04-01
categories: Regression
---
<br />
<h4>Example 1</h4>
Image of formula from <a href="https://en.wikipedia.org/wiki/Simple_linear_regression">Wikipedia</a>
<br />
<br />
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/ac3b42d4d7b7d8d496bbca97266021f73cceac84" alt="linear regression formula" />

{% highlight ruby %}
dataset = [[1, 1], [2, 3], [3, 2], [4, 3], [5, 5]]

x_mean = sum([row[0] for row in dataset])/float(len(dataset))
y_mean = sum([row[1] for row in dataset])/float(len(dataset))

covar = 0
for i in range(len(dataset)):
    x = dataset[i][0]
    y = dataset[i][1]
    covar_iteration = ((x - x_mean) * (y -y_mean)) / (len(dataset) - 1)
    covar += covar_iteration

using list comprehension: sum([(row[0] - x_mean) * (row[1] - y_mean) for row in dataset]) / (len(dataset) - 1)

var = 0
for i in range(len(dataset)):
    x = dataset[i][0]
    var_iteration = (x - x_mean)**2  / (len(dataset) - 1)
    var += var_iteration

using list comprehension: sum([(row[0]-x_mean)**2 for row in dataset]) / (len(dataset) - 1)

Bhat = covar / var
Ahat = y_mean - (Bhat * x_mean)

# See stats for sample correlation coefficient, uncorrected sample standard deviations, sample variance, and sample covariance
{% endhighlight %}

<br />
<h4>Example 2 Using Dot Product</h4>
<br />
Consider yhat = a + bx
<br />
<br />
Image of formula from <a href="http://www.statisticshowto.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation/">StatsHowTo</a>
<br />
<br />
<img src="http://www.statisticshowto.com/wp-content/uploads/2009/11/linearregressionequations.bmp" alt="linear regression formula" />
<p>Code credit: <a href="https://www.udemy.com/data-science-linear-regression-in-python">https://www.udemy.com/data-science-linear-regression-in-python</a>
{% highlight ruby %}
x_values = [row[0] for row in dataset]
y_values = [row[1] for row in dataset]

altX = np.array(x_values)
altY = np.array(y_values)

denominator = altX.dot(altX) - altX.mean() * altX.sum()
a = ( altX.dot(altY) - altY.mean() * altX.sum() ) / denominator
b = ( altY.mean() * altX.dot(altX) - altX.mean() * altX.dot(altY) ) / denominator
{% endhighlight %}

<br />
<h4>Example 3 Using Linear Algebra</h4>
<br />
<p>Code credit: <a href="https://www.kdnuggets.com/2016/11/linear-regression-least-squares-matrix-multiplication-concise-technical-overview.html">KDNuggets</a>
</p>
{% highlight ruby %}
x_stack = np.vstack(x_array)
x_stack = np.append(arr = np.ones((5, 1)).astype(int), values = x_stack, axis = 1)
print('w = ', np.linalg.inv(x_stack.T.dot(x_stack)).dot(x_stack.T).dot(Y_t))
or
print('w = ', np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y))
{% endhighlight %}
