---
layout: post
title: "Bag of Words"
date: 2020-06-19 12:30:00 
comments: false
categories: NLP
---

Bag of Words given freq_words, tokenize sentences dataset
```
import pandas as pd
import numpy as np
import nltk
import re

# Get the count for the most used words per sentence
X = []
for data in dataset:
    vector = []
    for i in range(len(freq_words)):
        if freq_words[i] in nltk.word_tokenize(data):
            vector.append(sum(1 for word in re.findall(r'\b[\w\']+\b', data) if word == freq_words[i]))
        else:
            vector.append(0)
    
    X.append(vector)

# Make a dataframe of the word count            
bow = pd.DataFrame(data=np.asarray(X), columns=freq_words)
print(bow.shape)
bow.head()
```