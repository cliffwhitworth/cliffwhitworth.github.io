---
layout: post
title: "Train Test Split"
date: 2020-07-23 19:11:00 
comments: false
categories: More
---

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop(['target', axis=1]), data['target'], test_size = 0.2)
```