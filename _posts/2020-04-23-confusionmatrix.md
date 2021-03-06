---
layout: post
title: "Confusion Matrix"
date: 2020-04-23 09:53:00 
comments: false
categories: More
---

* [Usage](https://nbviewer.jupyter.org/github/cliffwhitworth/machine_learning_notebooks/blob/master/DecisionTree.ipynb)

Create and plot a confusion matrix

```
# Confusion matrix with seaborn heatmap
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn

cm = confusion_matrix(y_test, predictions)
print(cm)
print()
cr = classification_report(y_test, predictions)
print(cr)

sn.heatmap(cm, annot=True, cmap='BuPu', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
