# Label Models

[![Build Status](https://travis-ci.com/BatsResearch/labelmodels.svg?token=sinAgJjnTsxQ2oN3R9vi&branch=master)](https://travis-ci.com/BatsResearch/labelmodels)

Lightweight implementations of generative label models for weakly supervised machine learning

# Example Usage - Naive Bayes Model
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
from labelmodels import NaiveBayes

# We initialize the model by specifying that there are 2 classes (binary
# classification) and 5 label sources
model = NaiveBayes(num_classes=2, num_lfs=5)

# Next, we estimate the model's parameters
model.estimate_label_model(votes)
print(model.get_accuracies())

# We can obtain a posterior distribution over the true labels
labels = model.get_label_distribution(votes)
```



# Example Usage - Partial Label Model
```python
# Let votes be an m x n matrix where m is the number of data examples, n is the
# number of label sources, and each element is in the set {-1, 0, 1, ..., k_l-1}, where
# k_l-1 is the number of label partitions for partial labeling functions PLF_{l}. If votes_{ij} is -1, it means that partial label source j
# abstains from voting on example i.

# As an example, we create a random votes matrix for classification with
# 1000 examples and 3 label sources
import numpy as np
import torch
simple_labelpartition_cfg = {
        0: [[1], [2, 3]],
        1: [[2], [1, 3]],
        2: [[3], [1, 2]]
}
num_sources = len(simple_labelpartition_cfg)
num_classes = 3
votes = np.random.randint(0, 1, size=(1000, 3))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# We now can create a Naive Bayes generative model to estimate the accuracies
# of these label sources
from labelmodels import PartialLabelModel
# We initialize the model by specifying that there are 2 classes (binary
# classification) and 5 label sources
model = PartialLabelModel(num_classes=num_classes,
                           labelpartition_cfg=simple_labelpartition_cfg,
                           preset_classbalance=None,
                           device=device)
# Next, we estimate the model's parameters
model.estimate_label_model(votes)
print(model.get_accuracies())

# We can obtain a posterior distribution over the true labels
labels = model.get_label_distribution(votes)
```