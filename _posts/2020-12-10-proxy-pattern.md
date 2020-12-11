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
    def thing1(self): return 'thing1 in Something'
    def thing2(self): return 'thing2 in Something'
    
class SomethingElse:
    def thing3(self): return 'thing3 in SomethingElse'
    def thing4(self): return 'thing4 in SomethingElse'

class Proxy:
    def __init__(self, my_obj):
        self.__proxy = my_obj

    def __getattr__(self, attr):
        def some_method(*args, **kwargs):
            result = getattr(self.__proxy, attr)(*args, **kwargs)
            return result    
        return some_method

class Everything:        
    def __getattr__(self, thing):
        self.proxy = Proxy(Something()) if thing in dir(Something) else Proxy(SomethingElse())
        return getattr(self.proxy, thing)


everything = Everything()
print(everything.thing1())
print(everything.thing2())
print(everything.thing3())
print(everything.thing4())

# thing1 in Something
# thing2 in Something
# thing3 in SomethingElse
# thing4 in SomethingElse
```

Javascript
```
class Something {
    thing1() { return 'thing1 in Something' }
    thing2() { return 'thing2 in Something' }
}

class SomethingElse {
    thing3() { return 'thing3 in SomethingElse' }
    thing4() { return 'thing4 in SomethingElse' }
}

const something = new Something()
const somethingelse = new SomethingElse()

const everything = new Proxy(somethingelse, {
    get: function(target, property) {
        return target[property] || something[property]
    }
});

console.log(everything.thing1())
console.log(everything.thing2())
console.log(everything.thing3())
console.log(everything.thing4())

// thing1 in Something
// thing2 in Something
// thing3 in SomethingElse
// thing4 in SomethingElse 
```