---
layout: post
title: "Factory Method"
date: 2020-12-13 14:02:00 
comments: false
categories: Python
---

Python Design Patterns
```
# https://danluu.com/empirical-pl

class Python:
    def __init__(self):
        self.contexts = {'Programming Paradigm': 'Scripting', 'Compilation Class': 'Dynamic', 'Type Class': 'Strong'}
        
    def summary(self, context):
        return self.contexts.get(context, context)

class JavaScript:
    def __init__(self):
        self.contexts = {'Programming Paradigm': 'Scripting', 'Compilation Class': 'Dynamic', 'Type Class': 'Weak'}
        
    def summary(self, context):
        return self.contexts.get(context, context)

class CSharp:
    def __init__(self):
        self.contexts = {'Programming Paradigm': 'Procedural', 'Compilation Class': 'Static', 'Type Class': 'Strong'}
        
    def summary(self, context):
        return self.contexts.get(context, context)
    
def Factory(language='Python'):
    languages = {
        'Python': Python,
        'JavaScript': JavaScript,
        'CSharp': CSharp
    }
    
    return languages[language]()

python = Factory('Python')
javascript = Factory('JavaScript')
csharp = Factory('CSharp')

context = ['Programming Paradigm', 'Compilation Class', 'Type Class']
language = ['Python', 'JavaScript', 'CSharp']

for lang in language:
    print(f'{lang}')
    print(', '.join(f'{cntx}: {eval(lang.lower()).summary(cntx)}' for cntx in context))
    print()
```