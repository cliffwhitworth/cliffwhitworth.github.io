---
layout: post
title: "spaCy Matcher"
date: 2020-07-03 14:21:00 
comments: false
categories: NLP
---

spaCy Matcher
```
import nltk
import spacy
from spacy.matcher import Matcher

# Load model
nlp = spacy.load('en_core_web_sm', disable=['new', 'textcat'])
text = '''This was done on the third of July. 
        The third is the day before the fourth of July. 
        The day after the fourth is the fifth of July.
        Next month will be the third of August.'''

pattern = [ {'POS':{'IN':['NOUN', 'ADJ']}},
            {'POS':'ADP'},
            {'LOWER':'july'}]

matcher = Matcher(nlp.vocab)

# matcher ID, callback, pattern
matcher.add('FindDayOfJuly', None, pattern)

# Tokenize text to track each sentence
tokenized_sentence = nltk.sent_tokenize(text)

for sentence in tokenized_sentence: 
    doc = nlp(sentence) # Tokenize the sentence for spaCy 
    matches = matcher(doc)
    words = nltk.word_tokenize(sentence) # Alt to see word tags in sentence
    # https://stackoverflow.com/questions/29332851/what-does-nn-vbd-in-dt-nns-rb-means-in-nltk
    print(nltk.pos_tag(words))
    for id, start, end in matches:
        string_id = nlp.vocab.strings[id]
        print(f'\n{string_id}: {doc[start:end]}\n')
```