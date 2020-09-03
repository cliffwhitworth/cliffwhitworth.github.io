---
layout: post
title: "Decorators"
date: 2020-09-03 12:25:00 
comments: false
categories: Python
---

```
# Fred Baptiste Python Deep Dive
from functools import wraps

def memoize(fn):
    cache = dict()

    @wraps(fn)
    def cache_manager(*args):
        if args not in cache:
            cache[args] = fn(*args)

        return cache[args]

    return cache_manager

@memoize
def fib(n):
    return 1 if n < 3 else fib(n-1) + fib(n-2)
```