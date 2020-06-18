---
layout: post
title: "NLTK Stemming and Lemmatization"
date: 2020-06-16 13:37:00 
comments: false
categories: NLP
---

NLTK Stemming and Lemmatization

Stemming

```
import nltk
from nltk.stem import PorterStemmer

sentences = nltk.sent_tokenize(text)
stemmer = PorterStemmer()
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    sentences[i] = ' '.join([stemmer.stem(word) for word in words])
    
sentences
```

Lemmatization
```
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
for i in range(len(sentences)):
    words = nltk. word_tokenize(sentences[i])
    sentences[i] = ' '.join([lemmatizer.lemmatize(word) for word in words])
    
sentences
```