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

Wraps takes a function used in a decorator and adds the functionality of copying over the function name, docstring, arguments list, etc

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

Another Example
```
from functools import wraps

def my_decorator(fn):
    print('In my_decorator')
    @wraps(fn)
    def my_wrapper(*args, **kwargs):
        kwargs['Whirled'] = 'Peas'
        print('In my_wrapper')
        print(f'{fn.__name__} was passed to my_wrapper; args: {args} and kwargs: {kwargs}')
        print('my_wrapper joins the args and adds a key:value to kwargs')
        return fn(' '.join(args), kwargs)
    
    return my_wrapper

@my_decorator
def funk(*args, **kwargs):
    """funk returns *args and **kwargs after some processing by my_wrapper"""
    
    return f'funk returns {args} {kwargs}'
    
print(funk('Hello', 'World!', Mello='Word', Yellow='Bird'))
print('Function name:', funk.__name__)
print('Function doc string:', funk.__doc__)
from functools import wraps

def my_decorator(fn):
    print('In my_decorator')
    @wraps(fn)
    def my_wrapper(*args, **kwargs):
        kwargs['Whirled'] = 'Peas'
        print('In my_wrapper')
        print(f'{fn.__name__} was passed to my_wrapper; args: {args} and kwargs: {kwargs}')
        print('my_wrapper joins the args and adds a key:value to kwargs')
        return fn(' '.join(args), kwargs)
    
    return my_wrapper

@my_decorator
def funk(*args, **kwargs):
    """funk returns *args and **kwargs after some processing by my_wrapper"""
    
    return f'funk returns {args} {kwargs}'
    
print(funk('Hello', 'World!', Mello='Word', Yellow='Bird'))
print('Function name:', funk.__name__)
print('Function doc string:', funk.__doc__)

# In my_decorator
# In my_wrapper
# funk was passed to my_wrapper; args: ('Hello', 'World!') and kwargs: {'Mello': 'Word', 'Yellow': 'Bird', 'Whirled': 'Peas'}
# my_wrapper joins the args and adds a key:value to kwargs
# funk returns ('Hello World!', {'Mello': 'Word', 'Yellow': 'Bird', 'Whirled': 'Peas'}) {}
# Function name: funk
# Function doc string: funk returns *args and **kwargs after some processing by my_wrapper
```

Signature
```
from inspect import signature
from functools import wraps

# The Signature object represents the call signature of a callable object and its return annotation.

def my_decorator1(fn):
    def my_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return my_wrapper

def my_decorator2(fn):
    @wraps(fn)
    def my_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return my_wrapper

def funk(a, b, key1='value1', key2='value2'):
    pass

@my_decorator1
def foo(a, b, **kwargs):
    pass

@my_decorator2
def bar(a, b, key1='value1', key2='value2'):
    pass

print('signature without wraps:', signature(my_decorator1(funk)))
print('signature with wraps:', signature(my_decorator2(funk)))
print('signature with wraps:', signature(foo))
print('signature with wraps:', signature(bar))

# signature without wraps: (*args, **kwargs)
# signature with wraps: (a, b, key1='value1', key2='value2')
# signature with wraps: (*args, **kwargs)
# signature with wraps: (a, b, key1='value1', key2='value2')
```