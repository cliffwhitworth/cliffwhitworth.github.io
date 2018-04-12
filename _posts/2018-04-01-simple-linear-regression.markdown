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

print('Iterations for covar (numerator)')
covarn = 0
for i in range(len(dataset)):
    x = dataset[i][0]
    y = dataset[i][1]
    covar_iteration = ((x - x_mean) * (y -y_mean)) / (len(dataset) - 1)
    covarn += covar_iteration

print('Iterations for var (denominator)')
test_var = 0
for i in range(len(dataset)):
    x = dataset[i][0]
    var_iteration = (x - x_mean)**2  / (len(dataset) - 1)
    test_var += var_iteration

covar = sum([(row[0] - x_mean) * (row[1] - y_mean) for row in dataset]) / (len(dataset) - 1)
var = sum([(row[0]-x_mean)**2 for row in dataset]) / (len(dataset) - 1)

Bhat = covar / var
Ahat = y_mean - (Bhat * x_mean)

// from scipy import stats
// from statistics import variance

x_array = [row[0] for row in dataset]
y_array = [row[1] for row in dataset]

xy_stacked = np.stack((x_array, y_array))
print('Covariance: ', np.cov(xy_stacked)[0][1])
print('Statistics variance: ', variance(x_array))
print('Numpy variance (sample): ', np.var(x_array,ddof=1))
print('Numpy covariance: ', np.cov(xy_stacked)[0][1]/variance(x_array))
print ('Stats Pearson (Rxy): ', stats.pearsonr(x_array, y_array)[0])
print ('Numpy CorrCoef: ', np.corrcoef(x_array, y_array)[0,1])
print ('Sx: ', np.std(x_array, axis=0))
print ('Sy: ', np.std(y_array, axis=0))
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
<p>Code credit: <a href="https://www.kdnuggets.com/2016/11/linear-regression-least-squares-matrix-multiplication-concise-technical-overview.html">https://www.kdnuggets.com/2016/11/linear-regression-least-squares-matrix-multiplication-concise-technical-overview.html</a>
</p>
{% highlight ruby %}
x_stack = np.vstack(x_array)
x_stack = np.append(arr = np.ones((5, 1)).astype(int), values = x_stack, axis = 1)
print('w = ', np.linalg.inv(x_stack.T.dot(x_stack)).dot(x_stack.T).dot(Y_t))
{% endhighlight %}
