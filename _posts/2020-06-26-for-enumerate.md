---
layout: post
title: "For Loop Enumerate"
date: 2020-06-26 15:22:00 
comments: false
categories: Python
---

Strings, Lists, etc

```
s = 'hello world!'
i = 0

for c in s:
    print(i, c)
    i += 1

for i in len(s):
    print(i, s[i])

for i, c in enumerate(s):
    print(i, c)
```

Dictionaries

```
for k, v in dict.items():
    print(k, v)
```