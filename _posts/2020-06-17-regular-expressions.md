---
layout: post
title: "Regular Expressions"
date: 2020-06-17 12:48:00 
comments: false
categories: Python
---

Regular Expressions

Abbreviate words in a string including lookbehind
```
import re

words = 'as far as i know'
print(''.join(re.findall(r'^\w|(?<=[\s_-])[a-zA-Z]', words.upper())))
```

Using lookahead
```
words = 'words-with-hypens and without'
print(re.findall(r'\w+(?=\b)', words))
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

Pangram
```
sentence = 'the quick brown fox jumps over the lazy dog'
print(string.ascii_lowercase == ''.join(
    sorted(set(re.sub(r'[\d\\"!&@$%^&_,-.]', '', sentence.lower())))).strip())
```