---
layout: post
title:  "OneHotEncoder"
date:   2020-08-04
categories: More
---

```
from sklearn.preprocessing import OneHotEncoder

ohe =  OneHotEncoder(drop='first', sparse=False)
ohe.fit(X_train[['Pclass', 'Sex', 'Embarked']])

map_list = [[f'{k}_{i}' for i in v] for k, v in zip(['Pclass', 'Sex', 'Embarked'], 
                                                    [[l for l in a][1:] for a in ohe.categories_])]
encoded_labels = [e for f in map_list for e in f]

X_train = pd.concat([X_train.drop(['Pclass', 'Sex', 'Embarked'], axis=1), 
           pd.DataFrame(ohe.transform(X_train[['Pclass', 'Sex', 'Embarked']]), 
                        columns=encoded_labels, index=X_train.index)], axis=1)
```