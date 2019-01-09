from .label_model import LabelModel, LearningConfig, init_random
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

        Initializes labeling function accuracies using optional argument and all
        other model parameters uniformly.

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

        :param votes: m x n matrix in {0, ..., k}, where m is the sum of the
                      lengths of the sequences in the batch, n is the number of
                      labeling functions and k is the number of classes
        :param seq_starts: vector of length l of row indices in votes indicating
                           the start of each sequence, where l is the number of
                           sequences in the batch. So, votes[seq_starts[i]] is
                           the row vector of labeling function outputs for the
                           first element in the ith sequence
        :return: vector of length l, where element is the log-likelihood of the
                 corresponding sequence of outputs in votes
        """
        raise NotImplementedError

    def _get_regularization_loss(self):
        """Computes the regularization loss of the model:
        acc_prior * \|lf_accuracy - init_lf_accuracy\|

        :return: value of regularization loss
        """
        return self.acc_prior * torch.norm(self.lf_accuracy - self.init_lf_acc)

    def estimate_label_model(self, votes, seq_starts, config=None):
        """Estimates the parameters of the label model based on observed
        labeling function outputs.

        Note that a minibatch's size refers to the number of sequences in the
        minibatch.

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :param seq_starts: vector of length l of row indices in votes indicating
                           the start of each sequence, where l is the number of
                           sequences in the batch. So, votes[seq_starts[i]] is
                           the row vector of labeling function outputs for the
                           first element in the ith sequence
        :param config: optional LearningConfig instance. If None, initialized
                       with default constructor
        """
        if config is None:
            config = LearningConfig()

        # Initializes random seed
        init_random(config.random_seed)

        # Converts to CSR and integers to standardize input
        votes = sparse.csr_matrix(votes, dtype=np.int)
        seq_starts = np.ndarray(seq_starts, dtype=np.int)

        # TODO: shuffle sequences

        # Creates minibatches
        seq_start_batches = [np.array(
            seq_starts[i * config.batch_size: (i + 1) * config.batch_size, :],
            copy=True)
            for i in range(int(np.ceil(votes.shape[0] / config.batch_size)))
       ]

        vote_batches = []
        for seq_start_batch in seq_start_batches:
            vote_batches.append(
                sparse.coo_matrix(
                    votes[seq_start_batch[0]:seq_start_batch[-1]], copy=True
                )
            )

        batches = zip(vote_batches, seq_start_batches)
        self._do_estimate_label_model(batches, config)

    def get_label_distribution(self, votes, seq_starts):
        raise NotImplementedError

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

    def get_start_balance(self):
        """Returns the model's estimated class balance for the start of a
        sequence

        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that the first
                 element in an example sequence has that label
        """
        start_balance = self.start_balance.detach().numpy()
        p = np.exp(start_balance - np.max(start_balance))
        return p / p.sum()

    def get_transition_matrix(self):
        """Returns the model's estimated transition distribution from class
        label to class label in a sequence.

        :return: a k x k Numpy matrix, in which each element i, j is the
        probability p(c_{t+1} = j + 1 | c_{t} = i + 1)
        """
        transitions = self.transitions.detach().numpy()
        for i in range(transitions.shape[0]):
            transitions[i] = np.exp(transitions[i] - np.max(transitions[i]))
            transitions[i] = transitions[i] / transitions[i].sum()
