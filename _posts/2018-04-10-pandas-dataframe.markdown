---
layout: post
title:  "Pandas Dataframe"
date:   2018-04-10 04
categories: More
---
<br />
<h4>Pandas Dataframe</h4>
<br />
<a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html">
Documentation
</a>
<br />

{% highlight ruby %}
# Create a dataframe of the features and add the target
import pandas as pd

df = pd.DataFrame(Features)
df.columns = [Feature_Names]
df['Target'] = Target
print('Dataframe Head')
print(df.head())
{% endhighlight %}
