---
layout: post
title: "Suptitle and Tight_Layout"
date: 2020-04-10 09:53:00 
comments: false
categories: More
---

* [Documentation](https://matplotlib.org/3.1.1/tutorials/intermediate/tight_layout_guide.html)
* [Usage](https://nbviewer.jupyter.org/github/cliffwhitworth/machine_learning_notebooks/blob/master/EsigningLoan.ipynb)

Using Tight Layout to Position Suptitle

```
# plot_features contains the features to plot
import Matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 12))
for i in range(plot_features.shape[1]):
    plt.subplot(6, 3, i + 1)
    ax = plt.gca()
    ax.set_title(plot_features.columns.values[i])
    vals = np.size(plot_features.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
        
    plt.hist(plot_features.iloc[:, i], bins=vals)

plt.suptitle('Histograms of Features', fontsize=18)
plt.tight_layout(pad=0.4, rect=[0, 0.03, 1, 0.95])
plt.show()
```
rect=[left, bottom, right, top] in normalized (0, 1) figure coordinates




