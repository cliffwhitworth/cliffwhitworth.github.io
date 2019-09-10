---
layout: post
title:  "Checklist"
date:   2018-03-30 10
categories: More
---

<br />
<h4>Checklist</h4>
<ul>
<li>Is there an expert on the data</li>
<li>Supervised or Unsupervised</li>
<ul>
<li>Is there a target</li>
</ul>
<li>Regression or Classification</li>
<ul>
<li>Continuous or discrete values (class labels)</li>
<li>Predicting a value or identifying group memebership</li>
</ul>
<li>Feature Selection</li>
<ul>
<li>
{% highlight ruby %}
# Remove null features
df.dropna(how='all', axis='columns', inplace=True)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Remove null observations
df.isnull().sum()
df.drop(df.index[[null_rows]], inplace=True)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Check for data leakage with expert
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Check for features that have only one value
constant_features = [
    feat for feat in df.columns if len(df[feat].unique()) == 1
]
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Check for features that have only zero values but have null values
constant_features = [
    feat for feat in df.columns if len(df[feat].fillna(0).unique()) == 1
]
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Check for Quasi Constant Values, does not count null values
for col in df.columns.sort_values():
    if (len(df[col].unique()) < 4):
        print(df[col].value_counts())
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Check for duplicate features
duplicated_feat = []
for i in range(0, len(dataset.columns)):
    if i % 10 == 0:  # Keep track of the loop
        print('loop tracker', i)

    col_1 = dataset.columns[i]

    for col_2 in dataset.columns[i + 1:]:
        if dataset[col_1].equals(dataset[col_2]):
            duplicated_feat.append(col_2)
{% endhighlight %}
</li>
</ul>
<li>
{% highlight ruby %}
# Does target need engineering
# Check for other features that need preliminary engineering
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Separate dataset into train, validate, and test
# Good practice to select the features by examining only the training set to avoid overfit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=['target'], axis=1),
    df['target'],
    test_size=0.2)
{% endhighlight %}
</li>
<li>More feature selection</li>
<ul>
<li>Check for correlated features</li>
<li>Feature importance</li>
<li>Mutual information</li>
<li>SelectKBest, SelectPercentile</li>
<li>Fisher Score - Chi-Square</li>
<li>ANOVA</li>
<li>ROC / AUC</li>
<li>Coefficients (Lasso)</li>
<li>Selection by model</li>
</ul>
<li>Feature Engineering</li>
<ul>
<li>Missing Data / Complete Case Analysis</li>
<li>Convert percentages to numerics</li>
<li>Outliers</li>
<li>Imputation</li>
<li>Model comparison</li>
<li>Reduce feature labels</li>
<li>Check for rare values</li>
<li>One hot encoding</li>
<li>Weight of evidence</li>
<li>Collinearity</li>
<li>Regularization</li>
</ul>
<li>Regression Models</li>
<ul>
<li>
{% highlight ruby %}
# Linear Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Polynomial Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model = PolynomialFeatures(degree = 4)
X_poly = model.fit_transform(X_train)
model.fit(X_poly, y_train)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)
lin_reg.predict(model.fit_transform(X_test))
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Support Vector Regression
# https://scikit-learn.org/stable/modules/svm.html#regression
from sklearn import svm

model = svm.SVR()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Decision Tree Regression
# https://scikit-learn.org/stable/modules/tree.html#regression
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Random Forest Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 50)
regressor.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# LassoCV Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
from sklearn.linear_model import LassoCV

model = LassoCV(cv=5,normalize=True,alphas=[.0005])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# AdaBoost Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor(n_estimators=100,loss="linear",learning_rate=.005)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Ridge Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
from sklearn.linear_model import Rigde

model = Ridge(random_state=10,normalize=True,alpha=.001)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# ElasticNet Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
from sklearn.linear_model import ElasticNet

model = linear_model.ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
</ul>
<li>Classification Models</li>
<ul>
<li>
{% highlight ruby %}
# Logistic Regression
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# K-Nearest Neighbor Classification
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Support Vector Classification
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
from sklearn.svm import SVC

model = SVC(kernel = 'linear')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Kernel SVM Classification
# https://scikit-learn.org/stable/auto_examples/svm/plot_custom_kernel.html#sphx-glr-auto-examples-svm-plot-custom-kernel-py
from sklearn.svm import SVC

model = SVC(kernel = 'rbf')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Naive Bayes Classification
# https://scikit-learn.org/stable/modules/naive_bayes.html
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Decision Tree Classification
# https://scikit-learn.org/stable/modules/tree.html#classification
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Random Forest Classification
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
{% endhighlight %}
</li>
</ul>
<li>Deep Learning Models</li>
<ul>
<li>
{% highlight ruby %}
# Artificial Neural Network

{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Convolutional Nueral Network

{% endhighlight %}
</li>
<li>
{% highlight ruby %}
# Recurrent Neural Network

{% endhighlight %}
</li>
</ul>
<li>Metrics - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics</li>
<li>Regression Metrics</li>
<ul>
<li></li>
</ul>
<li>Classification Metrics</li>
<ul>
<li></li>
</ul>
</ul>
