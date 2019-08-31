---
layout: post
title:  "Convolutional Neural Networks"
date:   2018-05-08
categories: DeepLearning
---
<br />

<a href="https://keras.io/layers/convolutional/">
Convolutional Layers
</a>
<br />
<a href="https://keras.io/layers/pooling/">
Pooling
</a>
<br />
<a href="https://keras.io/layers/core/">
Dense and Flatten
</a>
<br />
<a href="https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/">
Intuitive Explanation
</a>
<br />
<a href="http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/">
CNN in 11 lines
</a>
<br />
<a href="http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/">
Tutorial for TensorFlow
</a>
<br />
<a href="http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/">
Sigmoid vs Softmax
</a>

{% highlight ruby %}

# Importing libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize
classifier = Sequential()

# First convolutional layer
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling or downsampling
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer and pooling
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten for the fully connected layers
classifier.add(Flatten())

# Fully connected layers (neural nets)
classifier.add(Dense(number_of_units, activation='relu'))
classifier.add(Dense(number_of_classes, activation='softmax'))

# Optimization algorithm, loss function, and evaluation metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

{% endhighlight %}

{% highlight ruby %}

import tensorflow as tf

# using MNIST

i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

{% endhighlight %}

{% highlight ruby %}

import tensorflow as tf

i = Input(shape=x_train[0].shape)
# x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
# x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
# x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
# x = Dropout(0.2)(x)

# x = GlobalMaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

{% endhighlight %}