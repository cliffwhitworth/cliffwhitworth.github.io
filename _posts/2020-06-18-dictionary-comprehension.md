---
layout: post
title: "Dictionary Comprehension"
date: 2020-06-18 15:24:00 
comments: false
categories: More
---

Dictionary Comprehension

Print items per line
```
{print(f'{key}:{value}') for (key,value) in dictionary.items()}
```

If
```
{key:value for (key,value) in dictionary.items() if value == condition1 if value == condition2}
```

If Else
```
{key:(this if value==condition else that) for (key,value) in dictionary.items()}
```