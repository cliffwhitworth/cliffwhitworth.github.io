---
layout: post
title: "Transposition Map"
date: 2020-07-15 13:40:00 
comments: false
categories: Python
---

```
import string

text = 'Hello World!'
key = 5

reordered = ''.join([list(string.ascii_lowercase)[i % 26] for i in range(key, 26 + key)])
transposition_map = str.maketrans(string.ascii_letters, reordered + reordered.upper())
print(text.translate(transposition_map))
```


Alternate to reordered<br />
source: <a href='https://docs.python.org/2/library/collections.html'>https://docs.python.org/2/library/collections.html</a>
```
from collections import deque

my_sequence = 'abcdefg'
my_sequence.rotate(-3)
```