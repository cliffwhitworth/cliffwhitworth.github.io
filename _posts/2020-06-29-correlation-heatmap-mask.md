---
layout: post
title: "Correlation Heatmap with Mask"
date: 2020-06-29 14:23:00 
comments: false
categories: More
---

```
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation
sns.set(style='whitegrid', font_scale=2)
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(36,20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap,  annot=True, fmt='.3f')
plt.tight_layout()
```