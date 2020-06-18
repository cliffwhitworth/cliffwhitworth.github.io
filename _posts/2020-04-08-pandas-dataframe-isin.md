---
layout: post
title: "Pandas.DataFrame.isin"
date: 2020-04-08 16:25:06
comments: false
categories: Pandas
---

* [Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html)
* [Usage](https://nbviewer.jupyter.org/github/cliffwhitworth/machine_learning_notebooks/blob/master/CensusPredictions.ipynb)

A way to see if a string is in any of the features

```
df.isin([' ?']).any(axis='rows')
```


