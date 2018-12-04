---
layout: post
title:  "Permutations and Combinations"
date:   2018-04-02 06
categories: Stats
---
<br />
<a href="https://docs.python.org/2/library/itertools.html">
Itertools
</a>

{% highlight ruby %}

# Product(A, B) returns the same as ((x,y) for x in A for y in B).
# Product(A, repeat=4) means the same as product(A, A, A, A).
product('ABCD', repeat=2)

permutations('ABCD', 2)
combinations('ABCD', 2)
combinations_with_replacement('ABCD', 2)

# Excel
# PERMUT(n,r)
# COMBN(n,r)
# PERMUTATIONA(n,r)
# COMBINA(n,r)

{% endhighlight %}
