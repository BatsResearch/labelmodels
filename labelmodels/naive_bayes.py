from .label_model import LabelModel, LearningConfig, init_random
import numpy as np
from scipy import sparse
import torch
from torch import nn


class NaiveBayes(LabelModel):
    """A generative label model that assumes that all labeling functions are
    conditionally independent given the true class label, i.e., the naive Bayes
    assumption.

    Proposed in: A. P. Dawid and A. M. Skene. Maximum likelihood
    estimation of observer error-rates using the EM algorithm.
    Journal of the Royal Statistical Society C, 28(1):20–28, 1979.

    Proposed for labeling functions in: A. Ratner, C. De Sa, S. Wu, D. Selsam,
    and C. Ré. Data programming: Creating large training sets, quickly. In
    Neural Information Processing Systems, 2016.
    """

    def __init__(self, num_classes, num_lfs, init_lf_acc=.75, acc_prior=.1,
                 entropy_prior=0.01, learn_class_balance=True):
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
        :param learn_class_balance: whether to estimate the distribution over
                                    target classes (True) or assume to be
                                    uniform (False)
        """
        super(LabelModel, self).__init__()

        # Converts init_lf_acc to log scale
        init_lf_acc = -1 * np.log(1.0 / init_lf_acc - 1) / 2

        # Initializes parameters
        init_param = torch.tensor(
            [[init_lf_acc] * num_classes for _ in range(num_lfs)])
        self.accuracy = nn.Parameter(init_param)
        self.propensity = nn.Parameter(torch.zeros([num_lfs]))
        self.class_balance = nn.Parameter(
            torch.zeros([num_classes]), requires_grad=learn_class_balance)

        # Saves state
        self.num_classes = num_classes
        self.num_lfs = num_lfs
        self.init_lf_acc = init_lf_acc
        self.acc_prior = acc_prior
        self.entropy_prior = entropy_prior

    def forward(self, votes):
        """Computes log likelihood of labeling function outputs for each
        example in the batch.

        For efficiency, this function prefers that votes is an instance of
        scipy.sparse.coo_matrix. You can avoid a conversion by passing in votes
        with this class.

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: 1-d tensor of length m, where each element is the
                 log-likelihood of the corresponding row in labels
        """
        jll = self._get_observation_likelihoods(votes)
        mll = torch.logsumexp(jll, dim=1)

        return mll

    def _get_observation_likelihoods(self, votes):
        if type(votes) != sparse.coo_matrix:
            votes = sparse.coo_matrix(votes)

        # Initializes class log-likelihood as a 1-d tensor of length k
        class_ll = self._get_norm_class_balance()

        # Initializes joint log-likelihood of votes and class as an m x k matrix
        jll = class_ll.unsqueeze(0).repeat(votes.shape[0], 1)

        # Initializes normalizing constants
        z_prop = self.propensity.unsqueeze(1)
        z_prop = torch.cat((z_prop, torch.zeros((self.num_lfs, 1))), dim=1)
        z_prop = torch.logsumexp(z_prop, dim=1)

        z_acc = self.accuracy.unsqueeze(2)
        z_acc = torch.cat((z_acc, -1 * self.accuracy.unsqueeze(2)), dim=2)
        z_acc = torch.logsumexp(z_acc, dim=2)

        # Subtracts normalizing constant for propensities from jll
        # (since it applies to all outcomes)
        jll -= torch.sum(z_prop)

        # Loops over votes and classes to compute joint log-likelihood
        for i, j, v in zip(votes.row, votes.col, votes.data):
            for k in range(self.num_classes):
                if v == (k + 1):
                    logp = self.propensity[j] + self.accuracy[j, k] - z_acc[j, k]
                    jll[i, k] += logp
                elif v != 0:
                    logp = self.propensity[j] - self.accuracy[j, k] - z_acc[j, k]
                    logp -= torch.log(torch.tensor(self.num_classes - 1.0))
                    jll[i, k] += logp

        return jll

    def _get_regularization_loss(self):
        """Computes the regularization loss of the model:
        acc_prior * \|accuracy - init_lf_accuracy\|

        :return: value of regularization loss
        """
        neg_entropy = 0.0
        norm_class_balance = self._get_norm_class_balance()
        exp_class_balance = torch.exp(norm_class_balance)
        for k in range(self.num_classes):
            neg_entropy += norm_class_balance[k] * exp_class_balance[k]
        entropy_prior = self.entropy_prior * neg_entropy

        acc_prior = self.acc_prior * torch.norm(self.accuracy - self.init_lf_acc)

        return entropy_prior + acc_prior

    def _get_norm_class_balance(self):
        return self.class_balance - torch.logsumexp(self.class_balance, dim=0)

    def estimate_label_model(self, votes, config=None):
        """Estimates the parameters of the label model based on observed
        labeling function outputs.

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :param config: optional LearningConfig instance. If None, initialized
                       with default constructor
        """
        if config is None:
            config = LearningConfig()

        # Initializes random seed
        init_random(config.random_seed)

        # Converts to CSR to standardize input
        votes = sparse.csr_matrix(votes, dtype=np.int)

        # Shuffles rows
        index = np.arange(np.shape(votes)[0])
        np.random.shuffle(index)
        votes = votes[index, :]

        # Creates minibatches
        batches = [(sparse.coo_matrix(
            votes[i * config.batch_size: (i + 1) * config.batch_size, :],
            copy=True),)
            for i in range(int(np.ceil(votes.shape[0] / config.batch_size)))
        ]

        self._do_estimate_label_model(batches, config)

    def get_label_distribution(self, votes):
        """Returns the posterior distribution over true labels given labeling
        function outputs according to the model

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: m x k matrix, where each row is the posterior distribution over
                 the true class label for the corresponding example
        """
        labels = np.ndarray((votes.shape[0], self.num_classes))
        jll = self._get_observation_likelihoods(votes).detach().numpy()
        for i in range(votes.shape[0]):
            e_temp = np.exp(jll[i, :] - np.max(jll[i, :]))
            labels[i, :] = e_temp / e_temp.sum()

        return labels

    def get_class_balance(self):
        """Returns the model's estimated class balance

        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that an example
                 has that label
        """
        return np.exp(self._get_norm_class_balance().detach().numpy())

    def get_accuracies(self):
        """Returns the model's estimated labeling function accuracies
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function correctly outputs
                 the true class label, given that it does not abstain
        """
        acc = self.accuracy.detach().numpy()
        return np.exp(acc) / (np.exp(acc) + np.exp(-1 * acc))

    def get_propensities(self):
        """Returns the model's estimated labeling function propensities, i.e.,
        the probability that a labeling function does not abstain
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function does not abstain
        """
        prop = self.propensity.detach().numpy()
        return np.exp(prop) / (np.exp(prop) + 1)
