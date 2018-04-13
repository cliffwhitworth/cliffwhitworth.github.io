---
layout: post
title:  "Multiple Linear Regression"
date:   2018-04-02
categories: Regression
---
<br />
<h4>Scikit-learn LinearRegression</h4>
<p>Code credit:
<br />
<a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
</a>
<br />
<a href="https://www.udemy.com/machinelearning/">
https://www.udemy.com/machinelearning/
</a>
</p>

{% highlight ruby %}
# create training and test linearregressionequations

# fit data to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# test set prediction results
regressor.predict(x_test)

{% endhighlight %}

<br />
<h4>Stats Model Ordinary Least Squares</h4>
<p>Code credit:
<br />
<a href="https://www.statsmodels.org/stable/index.html">
https://www.statsmodels.org/stable/index.html
</a>
<br />
<a href="http://www.statsmodels.org/dev/endog_exog.html">
http://www.statsmodels.org/dev/endog_exog.html
</a>
</p>

{% highlight ruby %}

# add the bias
np.append(arr = np.ones((len(x), 1)).astype(int), values = x, axis = 1)

# feature array
features = x[:, [0, 1, 2, 3]]

# import statsmodels.formula.api as sm
regressor = sm.OLS(endog = y, exog = features).fit()
regressor.summary()

{% endhighlight %}
