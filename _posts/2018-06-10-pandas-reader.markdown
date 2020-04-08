---
layout: post
title:  "Pandas Datareader"
date:   2018-06-10 05
categories: More
---
<br />
<h4>IEX Example</h4>
<br />
<a href="https://pandas-datareader.readthedocs.io/en/latest/">
Documentation
</a>
<br />

{% highlight ruby %}

import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime

start = datetime.strptime('2018-01-01', '%Y-%m-%d')
end = datetime.strptime('2018-10-19', '%Y-%m-%d')

stock = web.DataReader('TSLA','iex',start,end)
# print(stock['close'].reset_index().drop('date', axis=1).head())
stock_open = stock['open'].reset_index().drop('date', axis=1)
stock_close = stock['close'].reset_index().drop('date', axis=1)

plt.plot(stock_open.index, stock_open.values.flatten(), label='open')
plt.plot(stock_close.index, stock_close.values.flatten(), label='close')

plt.xlabel('days')
plt.ylabel('value')
plt.legend(loc='best');
plt.show()

{% endhighlight %}
