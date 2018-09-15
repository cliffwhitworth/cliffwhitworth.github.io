---
layout: post
title:  "Make_Regression for Simple Linear Regression"
date:   2018-09-15
categories: Regression
---
<br />
<h4>Simple Linear Regression Example</h4>

<a href="https://statistics.laerd.com/spss-tutorials/multiple-regression-using-spss-statistics.php">
Laerd
</a>

<h4>Make_Regression</h4>

<a href="http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html">
Scikit-learn
</a>

<a href="https://github.com/cliffwhitworth/regression-notebooks/blob/master/SimpleLinearRegression.ipynb">
Notebook
</a>

{% highlight ruby %}

# Create X y dataset
X, y, coef = make_regression(n_samples = 100, n_features = 1, n_targets = 1, noise = 25, coef = True)

{% endhighlight %}

<h4>Plot and visualize the data</h4>

{% highlight ruby %}

# Plot data
plt.scatter(X, y)
plt.plot(np.unique(X.flatten()), np.poly1d(np.polyfit(X.flatten(), y, 1))(np.unique(X.flatten())))
plt.grid(True)
plt.show()

{% endhighlight %}

<h4>Create Dataframe</h4>

{% highlight ruby %}

# Create a dataframe of the feature and add the target
df = pd.DataFrame(X)
df.columns = ['X']
df['y'] = y
print('Dataframe Head')
print(df.head())

{% endhighlight %}

<h4>Descriptive Stats</h4>

{% highlight ruby %}

# Print descriptive stats
print(df.describe())

{% endhighlight %}

<h4>Regression Stats</h4>

<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html">
SciPy
</a>

{% highlight ruby %}

# Calculate linear least-squares regression
slope, intercept, r_value, p_value, std_err = stats.linregress(X.ravel(), y.ravel())
print('slope: ', slope)
print('intercept: ', intercept)
print('r_value: ', r_value)
print('r_squared: ', r_value**2)
print('p_value: ', p_value)
print ('std_err: ', std_err)

{% endhighlight %}

<h4>OLS Model</h4>

<a href="http://www.statsmodels.org/dev/examples/notebooks/generated/formulas.html">
Statsmodels
</a>

<a href="https://www.learndatasci.com/tutorials/predicting-housing-prices-linear-regression-using-python-pandas-statsmodels/">
Towards Data Science
</a>

{% highlight ruby %}

# Create model
model = ols("y ~ X", data=df).fit()
model.summary()

{% endhighlight %}

<h4>Confidence Intervals</h4>

<a href="https://www.statsmodels.org/dev/examples/notebooks/generated/wls.html">
Statsmodels (WLS)
</a>

{% highlight ruby %}

# Retrieve our confidence interval values with wls_prediction_std

_, confidence_interval_lower, confidence_interval_upper = wls_prediction_std(model)

fig, ax = plt.subplots(figsize=(10,7))

# Plot data
ax.plot(X, y, 'o', label='data')

# Plot trend line
ax.plot(X, model.fittedvalues, 'g-.', label='OLS')

# Plot confidence interval
ax.plot(X, confidence_interval_upper, 'r-', label='Confidence Intervals')
ax.plot(X, confidence_interval_lower, 'r-')

# Plot legend
ax.legend(loc='best');

{% endhighlight %}

<h4>Regression Plots</h4>

<a href="https://www.statsmodels.org/dev/generated/statsmodels.graphics.regressionplots.plot_regress_exog.html">
Statsmodels
</a>

<a href="https://www.statsmodels.org/dev/endog_exog.html">
Endog vs Exog
</a>

<a href="https://www.learndatasci.com/tutorials/predicting-housing-prices-linear-regression-using-python-pandas-statsmodels/">
Towards Data Science (Regression Plots)
</a>

{% highlight ruby %}

# Regression plots
fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "X", fig=fig)

{% endhighlight %}

<h4>Probability Plot</h4>

<a href="https://pythonfordatascience.org/linear-regression-python/">
Probability Plot
</a>

{% highlight ruby %}

# Probability Plot
stats.probplot(model.resid, dist="norm", plot= plt)
plt.title("Residuals Q-Q Plot")
plt.show()

{% endhighlight %}

<h4>Assumptions</h4>

<a href="https://www.youtube.com/watch?v=iMdtTCX2Q70">
Youtube
</a>

<h4>Assumption of Independent Errors</h4>

<a href="http://www.biostathandbook.com/independence.html">
Assumption of Independence
</a>

<a href="https://pythonfordatascience.org/linear-regression-python/">
Durbin Watson
</a>

{% highlight ruby %}

# Assumption of independent errors
print(statsmodels.stats.stattools.durbin_watson(model.resid))

{% endhighlight %}

<h4>Assumption of Residual Normality</h4>

{% highlight ruby %}

# Assumption of normality of the residuals
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(model.resid)
print(lzip(name, test))

{% endhighlight %}

<h4>Assumption of Homoscedasticity</h4>

{% highlight ruby %}

# Assumption of homoscedasticity
name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test = sms.het_breuschpagan(model.resid, model.model.exog)
print(lzip(name, test))

{% endhighlight %}

<h4>Gradient Descent Example</h4>

{% highlight ruby %}

def cost_function(X, y, theta):

    return ((X.dot(theta) - np.vstack(y.T)) ** 2).sum()/(2 * y.size)

def gradientDescent(X, y, theta, alpha, num_iters):

    # Initialize values
    J_history = np.zeros((num_iters, 1))
    colors = ['r', 'g', 'b', 'y', 'c']
    j = 0
    print()
    print('Values for the line equation from the first several iterations of the gradient descent')

    for i in range(num_iters):       
        # beta = beta - alpha * (X.T.dot(X.dot(beta)-y)/m)
        theta = theta - alpha*(1.0/m) * X.T.dot(X.dot(theta) - np.vstack(y.T))

        # cost history    
        J_history[i] = cost_function(X, y, theta)

        if i < 30 and i % 6 == 0:
            # Show some thetas and costs in the line equation as it approaches best fit
            # Assuming convergence is before 30 iterations
            print ('Iteration {}: y = {:0.4f} + {:0.4f}x and the cost: {:0.4f}'.format(i, theta[0][0], theta[1][0], J_history[i][0]))
            plt.plot(Xcopy, theta[0][0] + theta[1][0] * Xcopy, '-', c=colors[j])
            j += 1

    return theta, J_history

# Andrew Ng's M&Ns
m, n = X.shape # observations, features

# Save original X
Xcopy = X.copy()
Xcopy = Xcopy.flatten()

# Reshape X and add bias
X = np.append(arr = np.ones((y.size, 1)).astype(int), values = X.reshape(y.size, 1), axis = 1)

# Plot equation lines based on gradient descent
plt.figure(figsize=(10,3))

plt.subplot(121)

# Plot data
# y = 0.1383 + 0.7234x
plt.scatter(Xcopy, y)

# plt.plot(Xcopy, a1 + b1 * Xcopy, 'r-', linewidth=3)
plt.plot(np.unique(Xcopy), np.poly1d(np.polyfit(Xcopy, y, 1))(np.unique(Xcopy)), 'k-')

plt.grid(True)
plt.title('Lines converging on best fit')
plt.xlabel('X')
plt.ylabel('y')

# Choose a learning rate
alpha = 0.1
num_iters = 1000

# Init weights and run gradient descent
# theta = np.zeros((X.shape[1], 1))
theta=[[0], [0]]
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Get slope and intercept
# denominator = y.size * sum(map(lambda x:x*x,X)) - X.sum()**2
# a = ((y.sum() * X.dot(X)) - (X.sum() * sum(X * y))) / denominator
# b = ((y.size * sum(X * y)) - (X.sum() * y.sum())) / denominator
# print()
# print ('y = {:0.4f} + {:0.4f}x'.format(a, b))

# Similar method to get slope and intercept
d = Xcopy.dot(Xcopy) - Xcopy.mean() * Xcopy.sum()
a1 = ( y.mean() * Xcopy.dot(Xcopy) - Xcopy.mean() * Xcopy.dot(y) ) / d
b1 = ( Xcopy.dot(y) - y.mean() * Xcopy.sum() ) / d
print()
print ('Via formula: y = {:0.4f} + {:0.4f}x'.format(a1, b1))

plt.subplot(122)

# Plot the graph
plt.plot(range(J_history.size), J_history, "-b", linewidth=2 )
plt.title('Convergence of J(\u03B8) Against Iteration')
# r'J($\theta$)'
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.xlim((0, 40))
plt.show(block=False)

{% endhighlight %}

<h4>Split Train and Test</h4>

{% highlight ruby %}

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xcopy.reshape(y.size,1), y, test_size = 1/5, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# The intercept coefficient
print('Intercept: ', regressor.intercept_)
print('Coefficients: ', regressor.coef_[0])
# The mean squared error mse
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The root mean squared error rmse
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
# The mean absolute error mae
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Training Set')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Test Set')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

{% endhighlight %}
