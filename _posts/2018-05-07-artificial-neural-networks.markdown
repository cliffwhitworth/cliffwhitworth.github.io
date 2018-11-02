---
layout: post
title:  "Artificial Neural Networks"
date:   2018-05-07
categories: DeepLearning
---
<br />

<a href="https://keras.io/models/sequential/">
Keras Sequential Model
</a>
<br />
<a href="https://keras.io/layers/core/">
Keras Dense Layers
</a>
<br />
<a href="https://www.digitaltrends.com/cool-tech/what-is-an-artificial-neural-network/">
Digital Trends
</a>
<br />
<a href="https://www.analyticsvidhya.com/blog/2014/10/ann-work-simplified/">
Analytic Vidhya
</a>

{% highlight ruby %}

# Encode
# Split
# Scale

# Importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize
classifier = Sequential()

# Input and hidden layers
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

# Output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Optimization algorithm, loss function, and evaluation metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

{% endhighlight %}
