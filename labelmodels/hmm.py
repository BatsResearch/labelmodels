from .label_model import LabelModel, LearningConfig
import numpy as np
from scipy import sparse
import torch
from torch import nn


class HMM(LabelModel):
    """A generative label model that treats a sequence of true class labels as a
    Markov chain, as in a hidden Markov model, and treats all labeling functions
    as conditionally independent given the corresponding true class label, as
    in a Naive Bayes model.

    Proposed for crowdsourced sequence annotations in: A. T. Nguyen, B. C.
    Wallace, J. J. Li, A. Nenkova, and M. Lease. Aggregating and Predicting
    Sequence Labels from Crowd Annotations. In Annual Meeting of the Association
    for Computational Linguistics, 2017.
    """

    def __init__(self, num_classes, num_lfs, init_lf_acc=.6, acc_prior=.01,
                 learn_start_balance=False):
        """Constructor.

        Initializes labeling function accuracies using kwarg and all other model
        parameters uniformly.

        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param num_lfs: number of labeling functions to model
        :param init_lf_acc: initial estimated labeling function accuracy, must
                            be a float in [0,1]
        :param acc_prior: strength of regularization of estimated labeling
        function accuracies toward their initial values
        :param learn_start_balance: whether to estimate the distribution over
                                    target classes for the start of a sequence
                                    (True) or assume to be uniform (False)
        """
        super(LabelModel, self).__init__()

        # Converts init_lf_acc to log scale
        init_lf_acc = -1 * np.log(1.0 / init_lf_acc - 1) / 2

        # Initializes parameters
        self.lf_accuracy = nn.Parameter(torch.tensor([init_lf_acc] * num_lfs))
        self.lf_propensity = nn.Parameter(torch.zeros([num_lfs]))
        self.start_balance = nn.Parameter(torch.zeros([num_classes]),
                                          requires_grad=learn_start_balance)
        self.transitions = nn.Parameter(torch.zeros([num_classes, num_classes]))

        # Saves state
        self.num_classes = num_classes
        self.num_lfs = num_lfs
        self.init_lf_acc = init_lf_acc
        self.acc_prior = acc_prior

    def forward(self, votes, seq_starts):
        """
        Computes log likelihood of sequence of labeling function outputs for
        each (sequence) example in batch.

        For efficiency, this function prefers that votes is an instance of
        scipy.sparse.coo_matrix. You can avoid a conversion by passing in votes
        with this class.

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :param seq_starts:
        :return: 1-d tensor of length m, where each element is the
                 log-likelihood of the corresponding row in labels
        """
        raise NotImplementedError

    def _get_regularization_loss(self):
        return self.acc_prior * torch.norm(self.lf_accuracy - self.init_lf_acc)

    def estimate_label_model(self, votes, config=None):
        if config is None:
            config = LearningConfig()

        batcher = list(sparse.coo_matrix(
            votes[i * config.batch_size: (i + 1) * config.batch_size, :])
                       for i in range(int(np.ceil(votes.shape[0] / config.batch_size))))

        self._do_estimate_label_model(batcher, config)

    def get_label_distribution(self, votes):
        labels = np.ndarray((votes.shape[0], self.num_classes))
        log_acc = self.lf_accuracy.detach().numpy()
        log_class_balance = self.class_balance.detach().numpy()

        temp = np.ndarray((self.num_classes,))
        for i in range(labels.shape[0]):
            temp[:] = log_class_balance

            for j in range(self.num_lfs):
                vote = votes[i, j]
                if vote != 0:
                    temp[vote - 1] += log_acc[j]

            # Softmax
            e_temp = np.exp(temp - np.max(temp))
            labels[i, :] = e_temp / e_temp.sum()

        return labels

    def get_accuracies(self):
        """Returns the model's estimated labeling function accuracies

        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function correctly outputs
                 the true class label, given that it does not abstain
        """
        return 1 / (1 + np.exp(-2 * self.lf_accuracy.detach().numpy()))

    def get_propensities(self):
        """Returns the model's estimated labeling function propensities, i.e.,
        the probability that a labeling function does not abstain

        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function does not abstain
        """
        accuracies = self.lf_accuracy.detach().numpy()
        propensities = self.lf_propensity.detach().numpy()
        score = np.exp(propensities + accuracies) \
                + np.exp(propensities - accuracies)
        return score / (score + 1)

    def get_class_balance(self):
        """Returns the model's estimated class balance

        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that an example
                 has that label
        """
        class_balance = self.class_balance.detach().numpy()
        p = np.exp(class_balance - np.max(class_balance))
        return p / p.sum()