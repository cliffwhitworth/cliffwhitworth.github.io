---
layout: post
title:  "Assumptions"
date:   2019-04-01 02
categories: Stats
---
<br />
<a href="https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/">
Analytics Vidhya
</a>
<br />
<a href="https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/">
Ziganto
</a>
<br />
<a href="http://www.statsmodels.org/dev/diagnostic.html">
Stats Model
</a>

<h4>Ordinary Least Squares</h4>

{% highlight ruby %}

import statsmodels.formula.api as ols
X_with_bias = np.append(arr = np.ones((len(X), 1)).astype(int), values = X, axis = 1)
result = ols.OLS(endog = np.array(y).flatten(), exog = X_with_bias).fit()
print(result.summary())

{% endhighlight %}
<br />
<h4>Variance Inflation Factor (Multicolinearity)</h4>
<a href="https://etav.github.io/python/vif_factor_python.html">
Ernest Tavares III
</a>
{% highlight ruby %}

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

{% endhighlight %}
<br />
<h4>Outliers</h4>
<a href="https://stackoverflow.com/questions/10231206/can-scipy-stats-identify-and-mask-obvious-outliers">
StackOverflow Answer
</a>
{% highlight ruby %}

from statsmodels.formula.api import ols
regression = ols("data ~ x", data=dict(data=y, x=x)).fit()
test = regression.outlier_test()
outliers = ((x[i],y[i]) for i,t in enumerate(test.icol(2)) if t < 0.5)
print ('Outliers: ', list(outliers))

{% endhighlight %}
<br />
<h4>Heteroscedasticity</h4>
<a href="http://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html">
Stats Model Examples
</a>

{% highlight ruby %}

from statsmodels.compat import lzip
import statsmodels.stats.api as sms
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(result.resid, result.model.exog)
lzip(name, test)

{% endhighlight %}
<br />
<h4>Linearity</h4>

<a href="http://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html">
Stats Model Examples
</a>

{% highlight ruby %}

name = ['t value', 'p value']
test = sms.linear_harvey_collier(results)
lzip(name, test)

{% endhighlight %}
<br />
<h4>Influence Tests</h4>

<a href="http://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html">
Stats Model Examples
</a>
