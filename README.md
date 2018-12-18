# Label Models

[![Build Status](https://travis-ci.com/BatsResearch/labelmodels.svg?token=sinAgJjnTsxQ2oN3R9vi&branch=master)](https://travis-ci.com/BatsResearch/labelmodels)

Lightweight implementations of generative label models for weakly supervised machine learning

# Example Usage
```python
# Let votes be an m x n matrix where m is the number of data examples, n is the
# number of label sources, and each element is in the set {0, 1, ..., k}, where
# k is the number of classes. If votes_{ij} is 0, it means that label source j
# abstains from voting on example i.

# As an example, we create a random votes matrix for binary classification with
# 1000 examples and 5 label sources
import numpy as np
votes = np.random.randint(0, 3, size=(1000, 5))

# We now can create a Naive Bayes generative model to estimate the accuracies
# of these label sources
from labelmodels.naive_bayes import NaiveBayes

# We initialize the model by specifying that there are 2 classes (binary
# classification) and 5 label sources
model = NaiveBayes(num_classes=2, num_lfs=5)

# Next, we estimate the model's parameters
model.estimate_label_model(votes)
print(model.get_accuracies())

# We can obtain a posterior distribution over the true labels
labels = model.get_label_distribution(votes)
```