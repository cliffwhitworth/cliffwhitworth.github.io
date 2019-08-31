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

{% highlight ruby %}

# autoregressive model

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

N = len(X)

# build the model
i = Input(shape=(T,))
x = Dense(1)(i)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(lr=0.1),
)

# train the model
r = model.fit(
  X[:-N//2], Y[:-N//2],
  epochs=80,
  validation_data=(X[-N//2:], Y[-N//2:]),
)

# Forecast future values (use only self-predictions for making future predictions)

validation_target = Y[-N//2:]
validation_predictions = []

# last train input
last_x = X[-N//2] # 1-D array of length T

while len(validation_predictions) < len(validation_target):
  p = model.predict(last_x.reshape(1, -1))[0,0] # 1x1 array -> scalar
  
  # update the predictions list
  validation_predictions.append(p)
  
  # make the new input
  last_x = np.roll(last_x, -1)
  last_x[-1] = p

{% endhighlight %}

{% highlight ruby %}

import tensorflow as tf

# autoregressive RNN model
i = Input(shape=(T, 1))
x = SimpleRNN(5, activation='relu')(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(lr=0.1),
)

# train the RNN
r = model.fit(
  X[:-N//2], Y[:-N//2],
  epochs=80,
  validation_data=(X[-N//2:], Y[-N//2:]),
)

{% endhighlight %}
