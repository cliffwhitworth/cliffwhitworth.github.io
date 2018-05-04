---
layout: post
title:  "Dimensionality Reduction"
date:   2018-04-01 05
categories: More
---
<br />
<h4>Kernel PCA</h4>

<a href="http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html">
Sklearn
</a>
<br />
<a href="https://www.kaggle.com/lambdaofgod/kernel-pca-examples">
Kaggle Examples
</a>

{% highlight ruby %}

  # Split the dataset
  # Feature scaling

  # Apply Kernel
  from sklearn.decomposition import KernelPCA
  # kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
  # X_kpca = kpca.fit_transform(X)
  # X_back = kpca.inverse_transform(X_kpca)
  # pca = PCA()
  # X_pca = pca.fit_transform(X)
  kpca = KernelPCA(n_components = 2, kernel = 'rbf')
  X_train = kpca.fit_transform(X_train)
  X_test = kpca.transform(X_test)

  # Fit Logistic Regression to the Training set

{% endhighlight %}

<br />
<h4>Linear Discriminant Analysis</h4>

<a href="http://scikit-learn.org/stable/modules/lda_qda.html">
Sklearn
</a>
<br />
<a href="https://machinelearningmastery.com/linear-discriminant-analysis-for-machine-learning/">
Linear Discriminant Analysis
</a>
<br />
<a href="https://elitedatascience.com/dimensionality-reduction-algorithms">
Supervised Feature Extraction
</a>

{% highlight ruby %}

  # Split the dataset
  # Feature scaling

  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

  lda = LDA(n_components = 2)
  # X_r2 = lda.fit(X, y).transform(X)
  X_train = lda.fit_transform(X_train, y_train)
  X_test = lda.transform(X_test)

  # Fit Logistic Regression to the Training set

{% endhighlight %}

<br />
<h4>Principal Component Analysis</h4>

<a href="http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py">
Sklearn
</a>
<br />
<a href="http://setosa.io/ev/principal-component-analysis/">
Principal Component Analysis
</a>
<br />
<a href="https://elitedatascience.com/dimensionality-reduction-algorithms">
Unsupervised Feature Extraction
</a>

{% highlight ruby %}

  # Split the dataset
  # Feature scaling

  from sklearn.decomposition import PCA

  pca = PCA(n_components=2)
  # X_r = pca.fit(X).transform(X)
  X = pca.fit_transform(X_train)
  X = pca.transform(X_test)
  explained_variance = pca.explained_variance_ratio_

  # Percentage of variance explained for each components
  print('explained variance ratio (first two components): %s'
        % str(pca.explained_variance_ratio_))

  # Fit Logistic Regression to the Training set

{% endhighlight %}
