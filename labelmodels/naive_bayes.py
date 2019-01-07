from .label_model import LabelModel, LearningConfig
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

    def __init__(self, num_classes, num_lfs, init_lf_acc=.6, acc_prior=.01,
                 learn_class_balance=False):
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
        :param learn_class_balance: whether to estimate the distribution over
                                    target classes (True) or assume to be
                                    uniform (False)
        """
        super(LabelModel, self).__init__()

        # Converts init_lf_acc to log scale
        init_lf_acc = -1 * np.log(1.0 / init_lf_acc - 1) / 2

        # Initializes parameters
        self.lf_accuracy = nn.Parameter(torch.tensor([init_lf_acc] * num_lfs))
        self.lf_propensity = nn.Parameter(torch.zeros([num_lfs]))
        self.class_balance = nn.Parameter(torch.zeros([num_classes]),
                                          requires_grad=learn_class_balance)

        # Saves state
        self.num_classes = num_classes
        self.num_lfs = num_lfs
        self.init_lf_acc = init_lf_acc
        self.acc_prior = acc_prior

    def forward(self, votes):
        """
        Computes log likelihood of labeling function outputs for each example
        in batch.

        For efficiency, this function prefers that votes is an instance of
        scipy.sparse.coo_matrix. You can avoid a conversion by passing in votes
        with this class.

        :param votes: m x n matrix in {0, ..., k}, where m is the batch size,
                      n is the number of labeling functions and k is the number
                      of classes
        :return: 1-d tensor of length m, where each element is the
                 log-likelihood of the corresponding row in labels
        """
        # Checks that votes are in COO format
        if type(votes) != sparse.coo_matrix:
            votes = sparse.coo_matrix(votes)

        # Initializes class log-likelihood as a 1-d tensor of length k
        class_ll = self.class_balance - torch.logsumexp(self.class_balance, dim=0)

        # Initializes joint log-likelihood of class and votes as an m x k matrix
        jll = class_ll.unsqueeze(0).repeat(votes.shape[0], 1)

        # Initializes repeatedly used values
        prop_plus_acc = self.lf_propensity + self.lf_accuracy
        prop_minus_acc = self.lf_propensity - self.lf_accuracy
        prop_minus_acc_scaled = prop_minus_acc - \
                                torch.log(torch.tensor(self.num_classes - 1.0))

        # Computes conditional log-likelihood normalizing constants and
        # subtracts them from joint log-likelihoods
        z = torch.cat((prop_plus_acc.unsqueeze(0),
                       prop_minus_acc.unsqueeze(0),
                       torch.zeros((1, self.num_lfs))), dim=0)
        z = torch.logsumexp(z, dim=0)
        jll -= torch.sum(z)

        # Loops over votes and classes to compute joint log-likelihood
        for i, j, v in zip(votes.row, votes.col, votes.data):
            for k in range(self.num_classes):
                if v == (k + 1):
                    jll[i, k] += prop_plus_acc[j]
                else:
                    jll[i, k] += prop_minus_acc_scaled[j]

        # Computes marginal log-likelihood for each example
        mll = torch.logsumexp(jll, dim=1)

        return mll

    def _get_regularization_loss(self):
        return self.acc_prior * torch.norm(self.lf_accuracy - self.init_lf_acc)

    def estimate_label_model(self, votes, config=None):
        if config is None:
            config = LearningConfig()

        batcher = list(sparse.coo_matrix(
            votes[i * config.batch_size: (i+1) * config.batch_size, :])
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
