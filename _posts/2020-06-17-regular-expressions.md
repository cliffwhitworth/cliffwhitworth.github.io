---
layout: post
title: "Regular Expressions"
date: 2020-06-17 12:48:00 
comments: false
---

Regular Expressions

Abbreviate words in a string
```
import re

words = 'as far as i know'
print(''.join(re.findall(r'^\w|(?<=[\s_-])[a-zA-Z]', words.upper())))
```

Using word boundaries
```
sentence = 'It\'s no use going back to yesterday, because I was a different person then.'
print(re.findall(r'\b[\w\']+\b', sentence.lower()))
```

Character replacement
```
re.sub(r'[!&@$%^&_,-]', ' ', sentence)
```