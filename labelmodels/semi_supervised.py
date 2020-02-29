from .label_model import LearningConfig, init_random
from .naive_bayes import NaiveBayes
import numpy as np
from scipy import sparse
import torch


class SemiSupervisedNaiveBayes(NaiveBayes):
    """A generative label model that assumes that all labeling functions are
    conditionally independent given the true class label, i.e., the naive Bayes
    assumption.

    Supports partially observed settings where some ground truth labels are available.

    Proposed in: A. P. Dawid and A. M. Skene. Maximum likelihood
    estimation of observer error-rates using the EM algorithm.
    Journal of the Royal Statistical Society C, 28(1):20–28, 1979.

    Proposed for labeling functions in: A. Ratner, C. De Sa, S. Wu, D. Selsam,
    and C. Ré. Data programming: Creating large training sets, quickly. In
    Neural Information Processing Systems, 2016.
    """

    def __init__(self, num_classes, num_lfs, init_acc=.9, acc_prior=0.0,
                 balance_prior=0.0, learn_class_balance=True):
        """Constructor.

        Initializes labeling function accuracies using optional argument and all
        other model parameters uniformly.

        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param num_lfs: number of labeling functions to model
        :param init_acc: initial estimated labeling function accuracy, must
                            be a float in [0,1]
        :param acc_prior: strength of regularization of estimated labeling
                          function accuracies toward their initial values
        :param learn_class_balance: whether to estimate the distribution over
                                    target classes (True) or assume to be
                                    uniform (False)
        """
        super().__init__(num_classes, num_lfs, init_acc, acc_prior,
                         balance_prior, learn_class_balance)

    def forward(self, obs_votes, obs_labels, unobs_votes):
        """Computes log likelihood of labeling function outputs for each
        example in the batch.

        For efficiency, this function prefers that obs_votes and unobs_votes are
        instances of scipy.sparse.coo_matrix. You can avoid a conversion by
        passing in votes with this class.

        :param obs_votes: m x n matrix in {0, ..., k}, where m is the batch size,
                          n is the number of labeling functions and k is the
                          number of classes
        :param obs_labels: length m vector in {0, ..., k}, where m is the batch
                           size and k is the number of classes. A label of 0
                           means that the ground truth is unobserved
        :param unobs_votes: m x n matrix in {0, ..., k}, where m is the batch
                            size, n is the number of labeling functions and k is
                            the number of classes
        :return: 1-d tensor of length m, where each element is the
                 log-likelihood of the corresponding row in labels
        """
        class_ll = self._get_norm_class_balance()

        # Handles observed ground truth
        obs_ll = self._get_observed_likelihoods(obs_votes, obs_labels, class_ll)

        # Handles unobserved ground truth
        conditional_ll = self._get_labeling_function_likelihoods(unobs_votes)
        joint_ll = conditional_ll + class_ll
        unobs_ll = torch.logsumexp(joint_ll, dim=1)

        return torch.cat((obs_ll, unobs_ll), 0)

    def estimate_label_model(self, votes, labels, config=None):
        """Estimates the parameters of the label model based on observed
        labeling function outputs.

        :param votes: m x n matrix in {0, ..., k}, where m is the dataset size,
                      n is the number of labeling functions and k is the number
                      of classes
        :param labels: length m vector in {0, ..., k}, where m is the dataset size
                       and k is the number of classes. A label of 0 means that
                       the ground truth is unobserved
        :param config: optional LearningConfig instance. If None, initialized
                       with default constructor
        """
        if config is None:
            config = LearningConfig()

        # Initializes random seed
        init_random(config.random_seed)

        # Converts to CSR to standardize input
        votes = sparse.csr_matrix(votes, dtype=np.int)

        batches = self._create_minibatches(
            votes, labels, config.batch_size, shuffle_rows=True)
        self._do_estimate_label_model(batches, config)

    def _create_minibatches(self, votes, labels, batch_size, shuffle_rows=False):
        if shuffle_rows:
            index = np.arange(np.shape(votes)[0])
            np.random.shuffle(index)
            votes = votes[index, :]
            labels = labels[index]

        # Creates initial minibatches
        batches = [(votes[i * batch_size: (i + 1) * batch_size, :],
                    labels[i * batch_size: (i + 1) * batch_size])
                   for i in range(int(np.ceil(votes.shape[0] / batch_size)))
        ]

        # Splits minibatches into observed and unobserved ground truth
        batches = [
            (sparse.coo_matrix(batches[i][0][batches[i][1] != 0], copy=True),
             batches[i][1][batches[i][1] != 0],
             sparse.coo_matrix(batches[i][0][batches[i][1] == 0], copy=True))
            for i in range(len(batches))
        ]

        return batches

    def _get_observed_likelihoods(self, obs_votes, obs_labels, class_ll):
        if type(obs_votes) != sparse.coo_matrix:
            obs_votes = sparse.coo_matrix(obs_votes)

        obs_ll = class_ll[obs_labels - 1]

        # Initializes normalizing constants
        z_prop = self.propensity.unsqueeze(1)
        z_prop = torch.cat((z_prop, torch.zeros((self.num_lfs, 1))), dim=1)
        z_prop = torch.logsumexp(z_prop, dim=1)

        z_acc = self.accuracy.unsqueeze(2)
        z_acc = torch.cat((z_acc, -1 * self.accuracy.unsqueeze(2)), dim=2)
        z_acc = torch.logsumexp(z_acc, dim=2)

        # Subtracts normalizing constant for propensities from cll
        # (since it applies to all outcomes)
        obs_ll -= torch.sum(z_prop)

        # Loops over votes and classes to compute conditional log-likelihood
        for i, j, v in zip(obs_votes.row, obs_votes.col, obs_votes.data):
            gt = obs_labels[i]
            if v == gt:
                logp = self.propensity[j] + self.accuracy[j, gt - 1] - z_acc[j, gt - 1]
                obs_ll[i] += logp
            elif v != 0:
                logp = self.propensity[j] - self.accuracy[j, gt - 1] - z_acc[j, gt - 1]
                logp -= torch.log(torch.tensor(self.num_classes - 1.0))
                obs_ll[i] += logp

        return obs_ll
