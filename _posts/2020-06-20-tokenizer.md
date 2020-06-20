---
layout: post
title: "Tokenizer"
date: 2020-06-20 16:30:00 
comments: false
categories: NLP
---

Tokenizer
```
import heapq
import nltk
import re

# Tokenize sentences
dataset = nltk.sent_tokenize(text)
for i in range(len(dataset)):
    dataset[i] = re.sub(r'[\W\s]+', ' ', dataset[i]).lower()

# Tokenize words per sentence and get word count
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
```