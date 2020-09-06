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

With parameters
```
from functools import wraps

def count_seq(cntr):
    print(f'{cntr}. in cont_seq function')
    def decorator(fn):
        nonlocal cntr
        cntr += 1
        print(f'{cntr}. in decorator function')
        cntr += 1
        @wraps(fn)
        def upper_case(*args, **kwargs):
            nonlocal cntr
            print(f'{cntr}. in upper_case function')
            cntr += 1    
            print(f'{cntr}. Original greet function: {fn(*args)}')
            cntr += 1
            return f'{cntr}. upper_case function result: {fn(*args).upper()}'
        
        return upper_case
    return decorator

@count_seq(1)
def greet(s='hello'):
    return s

print(greet('hello world!'))

# Outputs
# 1. in cont_seq function
# 2. in decorator function
# 3. in upper_case function
# 4. Original greet function: hello world!
# 5. upper_case function result: HELLO WORLD!
```