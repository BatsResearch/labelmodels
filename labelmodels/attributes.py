from .label_model import LabelModel, LearningConfig, init_random
import numpy as np
from scipy.special import comb, softmax
import torch
import torch.nn as nn


class AttributeLabelModel(LabelModel):
    """
    TODO
    """
    def __init__(self, num_classes, num_sources, init_acc=.8, acc_prior=0.025,
                 balance_prior=0.0, learn_class_balance=True):
        """Constructor.

        Initializes source accuracies with argument, specificities uniformly.

        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param num_sources: number of label sources to model
        :param init_acc: initial estimated source accuracy, must
                            be a float in [0,1]
        :param acc_prior: strength of regularization of estimated source
                          accuracies toward their initial values
        """
        super().__init__()

        # Converts init_acc to log scale
        init_acc = -1 * np.log(1.0 / init_acc - 1) / 2

        init_param = torch.tensor(
            [[init_acc] * num_classes for _ in range(num_sources)])
        self.accuracy = nn.Parameter(init_param)
        self.specificity = nn.Parameter(torch.zeros((num_sources, num_classes)))
        self.class_balance = nn.Parameter(
            torch.zeros([num_classes]), requires_grad=learn_class_balance)

        # Saves state
        self.num_classes = num_classes
        self.num_lfs = num_sources
        self.init_acc = init_acc
        self.acc_prior = acc_prior
        self.balance_prior = balance_prior

    def forward(self, votes):
        """Computes log likelihood of source outputs for each example
        in the batch.

        :param votes: m x n x p tensor in {0, 1}, where m is the batch size,
                      n is the number of sources and p is the number
                      of classes
        :return: 1-d tensor of length m, where each element is the
                 log-likelihood of the corresponding slice in votes
        """
        class_ll = self._get_norm_class_balance()
        conditional_ll = self._get_vote_likelihoods(votes)
        joint_ll = conditional_ll + class_ll
        return torch.logsumexp(joint_ll, dim=1)

    def estimate_label_model(self, votes, config=None):
        """Estimates the parameters of the label model based on observed
        source outputs.

        :param votes: m x n x p tensor in {0, 1}, where m is the number of
                      examples, n is the number of sources and p is the number
                      of classes
        :param config: optional LearningConfig instance. If None, initialized
                       with default constructor
        """
        if config is None:
            config = LearningConfig()

        # Initializes random seed
        init_random(config.random_seed)

        batches = self._create_minibatches(
            votes, config.batch_size, shuffle_slices=True)
        self._do_estimate_label_model(batches, config)

    def get_label_distribution(self, votes):
        """Returns the posterior distribution over true labels given source
        outputs according to the model

        :param votes: m x n x p tensor in {0, 1}, where m is the number of
                      examples, n is the number of sources and p is the number
                      of classes
        :return: m x p matrix, where each row is the posterior distribution over
                 the true class label for the corresponding example
        """
        labels = np.ndarray((votes.shape[0], self.num_classes))
        batches = self._create_minibatches(votes, 2048, shuffle_slices=False)

        offset = 0
        for votes, in batches:
            class_balance = self._get_norm_class_balance()
            vote_likelihood = self._get_vote_likelihoods(votes)
            jll = class_balance + vote_likelihood
            for i in range(votes.shape[0]):
                p = torch.exp(jll[i, :] - torch.max(jll[i, :]))
                p = p / p.sum()
                for j in range(self.num_classes):
                    labels[offset + i, j] = p[j]
            offset += votes.shape[0]

        return labels

    def get_most_probable_labels(self, votes):
        """Returns the most probable true labels given observed source outputs.

        :param votes: m x n x p tensor in {0, 1}, where m is the number of
                      examples, n is the number of sources and p is the number
                      of classes
        :return: 1-d Numpy array of most probable labels
        """
        return np.argmax(self.get_label_distribution(votes), axis=1)

    def get_accuracies(self):
        """Returns the model's estimated source accuracies
        :return: an n x p NumPy array with each element in [0,1], representing
                 the estimated probability that the corresponding source's
                 output includes the true class label, conditioned on the true
                 class and given that it does not abstain, i.e., output all
                 possible classes
        """
        acc = self.accuracy.detach().numpy()
        return np.exp(acc) / (np.exp(acc) + np.exp(-1 * acc))

    def get_specificities(self):
        """Returns the models estimated specificities (the probability that its
        output contains a particular number of positive votes)
        :return: an n x p NumPy array with each element in [0,1]
        """
        spec = self.specificity.detach().numpy()
        return softmax(spec, axis=1)

    def get_class_balance(self):
        """Returns the model's estimated class balance

        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that an example
                 has that label
        """
        return np.exp(self._get_norm_class_balance().detach().numpy())

    def _get_vote_likelihoods(self, votes):
        """
        Computes conditional log-likelihood of labeling function votes given
        class as an m x p matrix.

        :param votes: m x n x p tensor in {0, 1}, where m is the batch size,
                      n is the number of sources and p is the number
                      of classes
        :return: matrix of dimension m x p, where element is the conditional
                 log-likelihood of votes given class
        """
        # Initializes conditional log-likelihood of votes as an m x k matrix
        cll = torch.zeros(votes.shape[0], self.num_classes)

        # Initializes normalizing constants
        z_spec = torch.logsumexp(self.specificity, dim=1).sum()

        z_acc = self.accuracy.unsqueeze(2)
        z_acc = torch.cat((z_acc, -1 * self.accuracy.unsqueeze(2)), dim=2)
        z_acc = torch.logsumexp(z_acc, dim=2)

        # Subtracts normalizing constant for specificities from cll
        # (since it applies to all outcomes)
        cll -= z_spec

        # Loops over votes and classes to compute conditional log-likelihood
        for i in range(votes.shape[0]):
            for j in range(votes.shape[1]):
                # Adds probability for number of positive votes
                num_votes = votes[i, j, :].sum()
                cll[i] += self.specificity[j, num_votes - 1]

                if num_votes < self.num_classes:
                    logp = self.accuracy[j] - z_acc[j]
                    constant = comb(self.num_classes - 1, num_votes - 1)
                    logp -= torch.log(torch.tensor(constant))
                    pos_votes = votes[i, j, :].float() * logp

                    logp = -1 * self.accuracy[j] - z_acc[j]
                    constant = comb(self.num_classes - 1, num_votes)
                    logp -= torch.log(torch.tensor(constant))
                    neg_votes = (1 - votes[i, j, :]).float() * logp

                    cll[i] += pos_votes + neg_votes

        return cll

    def _create_minibatches(self, votes, batch_size, shuffle_slices=False):
        if shuffle_slices:
            index = np.arange(np.shape(votes)[0])
            np.random.shuffle(index)
            votes = votes[index, :, :]

        # Creates minibatches
        batches = [(torch.tensor(votes[i * batch_size: (i + 1) * batch_size, :, :]),)
                   for i in range(int(np.ceil(votes.shape[0] / batch_size)))
                   ]

        return batches

    def _get_regularization_loss(self):
        neg_entropy = 0.0
        norm_class_balance = self._get_norm_class_balance()
        exp_class_balance = torch.exp(norm_class_balance)
        for k in range(self.num_classes):
            neg_entropy += norm_class_balance[k] * exp_class_balance[k]
        entropy_prior = self.balance_prior * neg_entropy

        return super()._get_regularization_loss() + entropy_prior

    def _get_norm_class_balance(self):
        return self.class_balance - torch.logsumexp(self.class_balance, dim=0)
