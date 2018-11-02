---
layout: post
title:  "Recurrent Neural Networks"
date:   2018-05-09
categories: DeepLearning
---
<br />

<a href="https://keras.io/layers/recurrent/">
Recurrent Layers
</a>
<br />
<a href="https://keras.io/layers/core/">
Dense and Dropout
</a>
<br />
<a href="https://keras.io/layers/recurrent/#lstm">
LSTM
</a>
<br />
<a href="https://github.com/sagar448/Keras-Recurrent-Neural-Network-Python">
RNN Example
</a>
<br />
<a href="https://machinelearningmastery.com/stacked-long-short-term-memory-networks/">
Stacked Long Short Term Memory Networks
</a>
<br />
<a href="http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/">
Tutorial for TensorFlow
</a>
<br />
<a href="http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/">
Sigmoid vs Softmax
</a>

{% highlight ruby %}

# Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize
model = Sequential()

# Adding LSTM layer and some Dropout regularisation
model.add(LSTM(units = number_of_units, return_sequences = True, input_shape=(X.shape[1], X.shape[2])))

# Add dropout regularization
model.add(Dropout(0.2))

# Add more LSTM layers and dropout regularization

# Add output layer
model.add(Dense(y.shape[1], activation='softmax'))

# Optimization algorithm and loss function
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit
model.fit(X, y, epochs=5, batch_size = 32)

{% endhighlight %}
