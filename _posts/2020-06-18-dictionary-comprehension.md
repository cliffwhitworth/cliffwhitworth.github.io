---
layout: post
title: "Dictionary Comprehension"
date: 2020-06-18 15:24:00 
comments: false
categories: Python
---

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

Zip and nested lists
```
ohe_cat_map = dict(zip(['Pclass', 'Sex', 'Embarked'], [[l for l in a] for a in ohe.categories_]))
map_list = [[f'{k}_{i}' for i in v] for k, v in ohe_cat_map.items()]
print([e for f in map_list for e in f])
```