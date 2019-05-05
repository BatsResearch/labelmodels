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

    def __init__(self, num_classes, num_lfs, init_lf_acc=.9, acc_prior=.025):
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
        init_param = torch.tensor(
            [[init_lf_acc] * num_classes for _ in range(num_lfs)])
        self.lf_accuracy = nn.Parameter(init_param)
        self.lf_propensity = nn.Parameter(torch.zeros([num_lfs]))
        self.start_balance = nn.Parameter(torch.zeros([num_classes]))
        self.transitions = nn.Parameter(torch.zeros([num_classes, num_classes]))

        # Saves state
        self.num_classes = num_classes
        self.num_lfs = num_lfs
        self.init_lf_acc = init_lf_acc
        self.acc_prior = acc_prior

    def observation_likelihood(self, votes):
        """
        Computes conditional log-likelihood of votes given class as an
        m x k matrix.

        For efficiency, this function prefers that votes is an instance of
        scipy.sparse.coo_matrix. You can avoid a conversion by passing in votes
        with this class.

        :param votes: m x n matrix in {0, ..., k}, where m is the sum of the
                      lengths of the sequences in the batch, n is the number of
                      labeling functions and k is the number of classes
        :return: matrix of dimension m x k, where element is the conditional
                 log-likelihood of votes given class
        """
        if type(votes) != sparse.coo_matrix:
            votes = sparse.coo_matrix(votes)

        # Initializes joint log-likelihood of votes as an m x k matrix
        cll = torch.zeros(votes.shape[0], self.num_classes)

        # Initializes normalizing constants
        z_prop = self.lf_propensity.unsqueeze(1)
        z_prop = torch.cat((z_prop, torch.zeros((self.num_lfs, 1))), dim=1)
        z_prop = torch.logsumexp(z_prop, dim=1)

        z_acc = self.lf_accuracy.unsqueeze(2)
        z_acc = torch.cat((z_acc, -1 * self.lf_accuracy.unsqueeze(2)), dim=2)
        z_acc = torch.logsumexp(z_acc, dim=2)

        # Subtracts normalizing constant for propensities from cll
        # (since it applies to all outcomes)
        cll -= torch.sum(z_prop)

        # Loops over votes and classes to compute joint log-likelihood
        for i, j, v in zip(votes.row, votes.col, votes.data):
            for k in range(self.num_classes):
                if v == (k + 1):
                    logp = self.lf_propensity[j] + self.lf_accuracy[j, k] - z_acc[j, k]
                    cll[i, k] += logp
                elif v != 0:
                    logp = self.lf_propensity[j] - self.lf_accuracy[j, k] - z_acc[j, k]
                    logp -= torch.log(torch.tensor(self.num_classes - 1.0))
                    cll[i, k] += logp

        return cll

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
        jll = self.observation_likelihood(votes)
        # Normalize transition matrix
        nor_transitions = self.transitions - torch.logsumexp(self.transitions, dim=1).unsqueeze(1).repeat(1, self.num_classes)
        nor_start_balance = self.start_balance - torch.logsumexp(self.start_balance, dim=0)
        for i in range(0, votes.shape[0]):
            if i in seq_starts:
                jll[i, :] = jll[i, :] + nor_start_balance
            else:
                joint_class_pair = jll[i-1, :].clone().unsqueeze(1).repeat(1, self.num_classes) + nor_transitions
                marginal_current_class = torch.logsumexp(joint_class_pair, dim=0)
                jll[i, :] = jll[i, :] + marginal_current_class
        seq_ends = [x - 1 for x in seq_starts] + [votes.shape[0]-1]
        seq_ends.remove(-1)
        mll = torch.logsumexp(jll[seq_ends], dim=1)
        return mll

    def viterbi(self, votes, seq_starts):
        """
        Computes the most probable underlying sequence nodes given estimated parameters

        :param votes: m x n matrix in {0, ..., k}, where m is the sum of the
                      lengths of the sequences in the batch, n is the number of
                      labeling functions and k is the number of classes
        :param seq_starts: vector of length l of row indices in votes indicating
                           the start of each sequence, where l is the number of
                           sequences in the batch. So, votes[seq_starts[i]] is
                           the row vector of labeling function outputs for the
                           first element in the ith sequence
        :return: vector of length m, where element is the most likely predicted labels
        """

        jll = self.observation_likelihood(votes)
        nor_transitions = self.transitions - torch.logsumexp(self.transitions, dim=1).unsqueeze(1).repeat(1, self.num_classes)
        nor_start_balance = self.start_balance - torch.logsumexp(self.start_balance, dim=0)

        T = votes.shape[0]
        bt = torch.zeros([T, self.num_classes])
        for i in range(0, T):
            if i in seq_starts:
                jll[i, :] += nor_start_balance
            else: 
                p = jll[i-1, :].clone().unsqueeze(1).repeat(
                    1, self.num_classes) + nor_transitions
                jll[i, :] += torch.max(p, dim=0)[0]
                bt[i, :] = torch.argmax(p, dim=0)

        seq_ends = [x - 1 for x in seq_starts] + [votes.shape[0] - 1]
        res = []
        j = T-1
        while j >= 0:
            if j in seq_ends:
                res.append(torch.argmax(jll[j, :]).item())
            if j in seq_starts:
                j -= 1
                continue
            res.append(int(bt[j, res[-1]].item()))
            j -= 1
        res = [x + 1 for x in res] 
        res.reverse()
        return res
              
    def _get_regularization_loss(self):
        """Computes the regularization loss of the model:
        acc_prior * |lf_accuracy - init_lf_accuracy|

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
        seq_starts = np.array(seq_starts, dtype=np.int)

        # TODO: shuffle sequences

        # Creates minibatches
        seq_start_batches = [np.array(
            seq_starts[i * config.batch_size: ((i + 1) * config.batch_size + 1)],
            copy=True)
            for i in range(int(np.ceil(len(seq_starts) / config.batch_size)))
        ]
        seq_start_batches[-1] = np.concatenate((seq_start_batches[-1], [votes.shape[0]]))

        vote_batches = []
        for seq_start_batch in seq_start_batches:
            vote_batches.append(
                sparse.coo_matrix(
                    votes[seq_start_batch[0]:seq_start_batch[-1]], copy=True
                )
            )

        seq_start_batches = [x[:-1]-x[0] for x in seq_start_batches]

        batches = list(zip(vote_batches, seq_start_batches))
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
        prop = self.lf_propensity.detach().numpy()
        return np.exp(prop) / (np.exp(prop) + 1)

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
        transitions = np.copy(self.transitions.detach().numpy())
        for i in range(transitions.shape[0]):
            transitions[i] = np.exp(transitions[i] - np.max(transitions[i]))
            transitions[i] = transitions[i] / transitions[i].sum()
        return transitions 
