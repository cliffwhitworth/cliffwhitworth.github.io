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

{% highlight ruby %}

# regression with tensorflow 2.x
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'),
  tf.keras.layers.Dense(1)
])

opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt, loss='mse')
r = model.fit(X, y, epochs=100)

{% endhighlight %}

{% highlight ruby %}

# classification with tensorflow 2.x
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

{% endhighlight %}
