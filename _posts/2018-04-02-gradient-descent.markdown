---
layout: post
title:  "Gradient Descent"
date:   2018-04-02
categories: More
---
<br />
<h4>Batch Gradient Descent</h4>

<a href="http://ozzieliu.com/2016/02/09/gradient-descent-tutorial/">
Ozzie Liu's Example
</a>

{% highlight ruby %}

def gradientDescent(X, y, theta, alpha, num_iters):

    # Initialize values
    J_history = np.zeros((num_iters, 1))
    m = X.shape[0]

    for i in range(num_iters):       
        # beta = beta - alpha * (X.T.dot(X.dot(beta)-y)/m)
        theta = theta - alpha*(1.0/m) * X.T.dot(X.dot(theta) - np.vstack(Y.T))

        # cost history    
        J_history[i] = cost_function(X, y, theta)

    return theta, J_history

{% endhighlight %}
