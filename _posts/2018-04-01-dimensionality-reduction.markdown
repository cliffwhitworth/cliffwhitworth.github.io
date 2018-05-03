---
layout: post
title:  "Dimensionality Reduction"
date:   2018-04-01 05
categories: More
---
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
Unsupervised Feature Extraction
</a>

{% highlight ruby %}

  # Split the dataset
  # Feature scaling

  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

  lda = LDA(n_components = 2)
  # X_r2 = lda.fit(X, y).transform(X)
  X_train = lda.fit_transform(X_train, y_train)
  X_test = lda.transform(X_test)

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

{% endhighlight %}
