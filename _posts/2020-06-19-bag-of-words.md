---
layout: post
title: "Bag of Words"
date: 2020-06-19 12:30:00 
comments: false
categories: NLP
---

Bag of Words
```
import pandas as pd
import numpy as np
import heapq
import nltk
import re

# Break text into sentences
dataset = nltk.sent_tokenize(text)
for i in range(len(dataset)):
    dataset[i] = re.sub(r'[\W\s]+', ' ', dataset[i]).lower()

# Get word count
wordcount = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word in wordcount.keys():
            wordcount[word] += 1
        else:
            wordcount[word] = 1

# Get the 100 most used words
freq_words = heapq.nlargest(100, wordcount, key=wordcount.get)

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