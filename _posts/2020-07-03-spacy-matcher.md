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

Phrase Matcher

```
import bs4 as bs
import urllib.request
import re
import spacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import PhraseMatcher
pmatcher = PhraseMatcher(nlp.vocab)

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Alice%27s_Adventures_in_Wonderland').read()
soup = bs.BeautifulSoup(source, 'lxml')

text = ''
for paragraph in soup.find_all('p'):
    text += paragraph.text
    
text = re.sub(r'\[[0-9]+\]', ' ', text.lower()).strip()
text = re.sub(r'[\n\s]+', ' ', text)

txt = nlp(text)

phrase_list = ['white rabbit', 'blue caterpillar', 'mock turtle', 'march hare', 'cheshire cat']
phrase_patterns = [nlp(p) for p in phrase_list]
pmatcher.add('AliceWonders', None, *phrase_patterns)
matches_found = pmatcher(txt)

matches_list = []
for id, start, end in matches_found:
    name = nlp.vocab.strings[id]
    span = txt[start:end]
    matches_list.append(span.text)
    print(start, end, span.text)

print()
sentences = [s for s in txt.sents]
for s in sentences:
    s_joined = ''.join([token.text_with_ws for token in s])
    for m in matches_list:        
        if m in s_joined:
            print(s_joined.strip())
            break
```