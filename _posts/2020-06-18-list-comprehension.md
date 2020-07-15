---
layout: post
title: "List Comprehension"
date: 2020-06-18 13:03:00 
comments: false
categories: Python
---

List Comprehension

Sum
```
sum([value for value in list_of_values])
```

If
```
[item for item in item_list if item == 'something']
```

Map
```
def foo(bar):
    return [x for x in map(dictionary.get, list[bar])]
    # [dictionary[i] for i in object]
```

Reorder
```
[list(string.ascii_lowercase)[i % 26] for i in range(start_at_index, 26 + start_at_index)]
```

Alternate to reordered<br />
source: <a href='https://docs.python.org/2/library/collections.html'>https://docs.python.org/2/library/collections.html</a>
```
from collections import deque

my_sequence = 'abcdefg'
my_sequence.rotate(-3)
```