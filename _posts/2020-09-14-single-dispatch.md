---
layout: post
title: "Single Dispatch"
date: 2020-09-14 14:11:00 
comments: false
categories: Python
---

Without single dispatch
```
# Fred Baptiste Python Deep Dive
from html import escape

s = """this is 
a multi line string
with special characters: 10 < 100"""

def html_default(arg):
    return escape(str(arg))

def html_str(arg):
    return html_default(html_default(arg)).replace('\n', '<br />\n')

print(html_str(s))
```

Using singledispatch
```
from functools import singledispatch

@singledispatch
def encode_html(arg):
    return escape(str(arg))

@encode_html.register(str)
def html_str(arg):
    return escape(arg).replace('\n', '<br />\n')

encode_html.dispatch(str)

encode_html(s)
```