---
layout: post
title: "Proxy Pattern"
date: 2020-12-10 16:55:00 
comments: false
categories: Python
---

Proxy Pattern in Python
```
class Something:    
    def func1(self): return 'func1 in Something'
    def func2(self): return 'func2 in Something'
    
class SomethingElse:
    def func3(self): return 'func3 in SomethingElse'
    def func4(self): return 'func4 in SomethingElse'

class Proxy:
    def __init__(self, my_obj):
        self.__proxy = my_obj

    def __getattr__(self, attr):
        def some_method(*args, **kwargs):
            result = getattr(self.__proxy, attr)(*args, **kwargs)
            return result    
        return some_method

class Everything:        
    def __getattr__(self, func):
        self.proxy = Proxy(Something()) if func in dir(Something) else Proxy(SomethingElse())
        return getattr(self.proxy, func)


everything = Everything()
print(everything.func1())
print(everything.func2())
print(everything.func3())
print(everything.func4())

# func1 in Something
# func2 in Something
# func3 in SomethingElse
# func4 in SomethingElse
```

Javascript
```
class Something {
    func1() { return 'func1 in Something' }
    func2() { return 'func2 in Something' }
}

class SomethingElse {
    func3() { return 'func3 in SomethingElse' }
    func4() { return 'func4 in SomethingElse' }
}

const something = new Something()
const somethingelse = new SomethingElse()

const everything = new Proxy(somethingelse, {
    get: function(target, property) {
        return target[property] || something[property]
    }
});

console.log(everything.func1())
console.log(everything.func2())
console.log(everything.func3())
console.log(everything.func4())

// func1 in Something
// func2 in Something
// func3 in SomethingElse
// func4 in SomethingElse
```