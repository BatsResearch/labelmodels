from .label_model import LabelModel, init_random, LearningConfig
import numpy as np
from scipy import sparse
import torch
from torch import nn
from copy import deepcopy as dc
import logging


class PartialLabelLearningConfig(LearningConfig):
    """Container for hyperparameters used by PartialLabelModel during learning"""
    def __init__(self):
        """Initializes all hyperparameters to default values"""
        super().__init__()
        self.epochs = 200
        self.batch_size = 8192
        self.step_size = 0.1
        self.step_schedule = 'p'
        self.step_size_mult = 0.1
        self.momentum = 0.8


class PartialLabelModel(LabelModel):
    """A generative label model that assumes that all partial labeling functions are
    conditionally independent given the true class label. A naive Bayes distribution
    is assumed.
    """

    def __init__(self, num_classes,
                 label_partition,
                 init_acc=0.7, preset_classbalance=None,
                 learn_class_balance=True,
                 device='cpu'):
        """Constructor.

        Initializes labeling function accuracies using optional argument and all
        other model parameters uniformly.

        :param num_classes: number of target classes, i.e., binary
                            classification = 2
        :param label_partition: partial labeling functions configurations. The label_partition configures the label
                                partitions mapping in format as
                                {PLF's index: [partition_1, partition_2, ..., partition_{k_l}]}
        :param preset_classbalance: None if want to learn class balance. Can be preset as fixed class balance
        :param init_acc: initial estimated labeling and linking function
                         accuracy, must be a float in [0,1]
        :param device: calculation device
        """
        super().__init__()

        self.device = device
        if not torch.cuda.is_available():
            self.device = 'cpu'
        self.preset_classbalance = preset_classbalance
        self.num_classes = num_classes
        self.init_acc = -1 * np.log(1.0 / init_acc - 1) / 2
        self.label_partition = label_partition
        self.num_df = len(label_partition)

        if self.preset_classbalance is not None:
            self.class_balance = torch.nn.Parameter(
                torch.log(self.preset_classbalance),
                requires_grad=False
            )
        else:
            self.class_balance = torch.nn.Parameter(
                torch.zeros([self.num_classes], device=self.device),
                requires_grad=learn_class_balance
            )

        self.accuracy = torch.nn.Parameter(
            torch.ones([self.num_df, self.num_classes], device=self.device) * self.init_acc,
            requires_grad=True
        )

        self.propensity = torch.nn.Parameter(
            torch.zeros([self.num_df], device=self.device),
            requires_grad=True
        )

        self.ct = torch.zeros([self.num_df, self.num_classes])
        self.poslib = torch.zeros([self.num_df, self.num_classes])
        self.neglib = torch.zeros([self.num_df, self.num_classes])

        '''
        Set Ops
        '''
        def intercect(l1, l2):
            return [value for value in l1 if value in l2]

        def union(l1, l2):
            return list(set(l1) | set(l2))

        for fid, clusters in self.label_partition.items():
            crange = clusters[0]
            ccover = []
            for cluster_id, cluster in enumerate(clusters):
                cluster.sort()
                self.label_partition[fid][cluster_id] = cluster
                crange = intercect(crange, cluster)
                ccover = union(ccover, cluster)
            if len(crange) > 0:
                raise RuntimeError('Setup Violation: No class can appear in all groups!')
            if len(ccover) < self.num_classes:
                raise RuntimeError('Setup Violation: Class must appear at least once! Please setup a dummy label group if necessary!')

        for fid, clusters in self.label_partition.items():
            for cluster_id, cluster in enumerate(clusters):
                for class_id in cluster:
                    self.poslib[fid, class_id - 1] += 1
                    self.ct[fid, class_id - 1] = cluster_id
            self.neglib[fid, :] = len(clusters) - self.poslib[fid, :]
        self.poslib[self.poslib == 0] = 1

    def forward(self, votes, bid):
        """Computes log likelihood of labeling function outputs for each
        example in the batch.
        """
        class_ll = self._get_norm_class_balance()
        conditional_ll = self._cll(votes, bid)
        joint_ll = conditional_ll + class_ll
        return torch.logsumexp(joint_ll, dim=1)

    def estimate_label_model(self, votes, config=None):
        """Estimates the parameters of the label model based on observed
        labeling function outputs.
        """
        if config is None:
            config = PartialLabelLearningConfig()

        # Initializes random seed
        init_random(config.random_seed)

        batches = self._setup(votes, config.batch_size, shuffle=True)

        self._do_estimate_label_model(batches, config)

    def get_label_distribution(self, votes, annot_batch_sz=2048):
        """Returns the posterior distribution over true labels given labeling
        function outputs according to the model

        :param votes: m x n matrix where each element is in the set {0, 0, 1, ..., k_l}, where
                      k_l is the number of label partitions for partial labeling functions PLF_{l}.
        :return: m x k matrix, where each row is the posterior distribution over
                 the true class label for the corresponding example
        """
        self.eval()
        batches = self._setup(votes, annot_batch_sz)

        labels = np.ndarray((votes.shape[0], self.num_classes))
        for batch_id, batch_votes in enumerate(batches):
            class_balance = self._get_norm_class_balance()
            lf_likelihood = self._cll(batch_votes, batch_id)
            jll = class_balance + lf_likelihood
            P = torch.exp(jll - torch.max(jll, dim=1)[0].unsqueeze(1).repeat(1, self.num_classes))
            P /= torch.sum(P, dim=1).unsqueeze(1).repeat(1, self.num_classes)
            labels[batch_id*annot_batch_sz:batch_id*annot_batch_sz+batch_votes.shape[0]] = P.detach().cpu().numpy()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        return labels

    def get_most_probable_labels(self, votes):
        """Returns the most probable true labels given observed function outputs.

        :param votes: m x n matrix where each element is in the set {0, 0, 1, ..., k_l}, where
                      k_l is the number of label partitions for partial labeling functions PLF_{l}.
        :return: 1-d Numpy array of most probable labels
        """
        return np.argmax(self.get_label_distribution(votes), axis=1) + 1

    def get_class_balance(self):
        """Returns the model's estimated class balance

        :return: a NumPy array with one element in [0,1] for each target class,
                 representing the estimated prior probability that an example
                 has that label
        """
        return np.exp(self._get_norm_class_balance().detach().cpu().numpy())

    def get_accuracies(self):
        """Returns the model's estimated labeling function accuracies
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function correctly outputs
                 the true class label, given that it does not abstain
        """
        acc = self.accuracy.detach().cpu().numpy()
        return np.exp(acc) / (np.exp(acc) + np.exp(-1 * acc))

    def get_propensities(self):
        """Returns the model's estimated labeling function propensities, i.e.,
        the probability that a labeling function does not abstain
        :return: a NumPy array with one element in [0,1] for each labeling
                 function, representing the estimated probability that
                 the corresponding labeling function does not abstain
        """
        prop = self.propensity.detach().cpu().numpy()
        return np.exp(prop) / (np.exp(prop) + 1)

    def _do_estimate_label_model(self, batches, config):
        """Internal method for optimizing model parameters.

        :param batches: sequence of inputs to forward(). The sequence must
                        contain tuples, even if forward() takes one
                        argument (besides self)
        :param config: an instance of PartialLabelLearningConfig
        """

        optimizer = torch.optim.Adam(
            self.parameters(), lr=config.step_size,
            weight_decay=0)

        if config.step_schedule == 'p':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-10, factor=config.step_size_mult)
        elif config.step_schedule == 'c':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-1, max_lr=0.2)
        elif config.step_schedule is not None and config.step_size_mult is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.step_schedule, gamma=config.step_size_mult)
        else:
            scheduler = None

        self.train()

        for epoch in range(config.epochs):
            ga = dc(self.accuracy)
            logging.info('Epoch {}/{}'.format(epoch + 1, config.epochs))
            running_loss = 0.0
            epoch_loss = []
            for i_batch, inputs in enumerate(batches):
                optimizer.zero_grad()
                log_likelihood = self(inputs, i_batch)
                loss = -1 * torch.mean(log_likelihood)
                loss += self._get_regularization_loss()
                loss.backward()
                optimizer.step()
                running_loss += loss
                epoch_loss.append(float(loss.item()))
            epoch_loss = running_loss / len(batches)
            logging.info('Train Loss: %.6f', epoch_loss)
            if torch.sum(torch.abs(self.accuracy - ga)) < 1e-7:
                logging.info('1e-7 Criterion Reached: Epoch')
                break
            if scheduler is not None:
                if config.step_schedule == 'p':
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

        if 'cuda' in self.device:
            torch.cuda.empty_cache()

    def _setup(self, votes, batch_size, shuffle=False):
        ''' Setup \& precalculates/populates helper variables

        :param votes: Full PLFs votes input.
        :param batch_size: # of instances in one batch
        :param shuffle_rows: Decides if rows of given votse need to shuffle

        :return: 3-d Numpy array for batched votes in shape [# batch, # instance, # plfs]
        '''
        # Normalizing to 0-indexed LPs.
        batches = self._create_minibatches(votes-1, batch_size, shuffle)
        cth = self.ct.unsqueeze(0).repeat(batch_size, 1, 1)
        self.c = torch.zeros([len(batches), batch_size, self.num_df, self.num_classes])
        self.n = torch.zeros([len(batches), batch_size, self.num_df, self.num_classes])
        self.a = torch.ones([len(batches), batch_size, self.num_df, self.num_classes])
        self.p = torch.ones([len(batches), batch_size, self.num_df])
        for bid in range(len(batches) - 1):
            extb = batches[bid].unsqueeze(2).repeat(1, 1, self.num_classes)
            self.c[bid] = torch.where(torch.eq(cth, extb), torch.tensor(1.0), torch.tensor(-1.0))
            self.a[bid] = torch.where(extb==-1, torch.tensor(0.0), torch.tensor(1.0))
            marker = torch.where(self.c[bid]==1, torch.tensor(1.0), torch.tensor(0.0))
            self.n[bid] = (1 - marker) * self.neglib + marker * self.poslib
            self.p[bid] = torch.where(batches[bid]==-1, torch.tensor(0.0), torch.tensor(1.0))

        last_bz = len(batches[-1])
        last_extb = batches[-1].unsqueeze(2).repeat(1, 1, self.num_classes)
        self.c[-1, :last_bz] = torch.where(
            torch.eq(cth[:last_bz, :, :], last_extb),
            torch.tensor(1.0), torch.tensor(-1.0))
        marker = torch.where(self.c[-1, :last_bz] == 1, torch.tensor(1.0), torch.tensor(0.0))
        self.a[-1, :last_bz] = torch.where(last_extb==-1, torch.tensor(0.0), torch.tensor(1.0))
        self.n[-1, :last_bz] = (1 - marker) * self.neglib + marker * self.poslib
        self.n = -torch.log(self.n)
        self.p[-1, :last_bz] = torch.where(batches[-1]==-1, torch.tensor(0.0), torch.tensor(1.0))
        return batches

    def _get_regularization_loss(self):
        """Gets the value of the regularization loss for the current values of
        the model's parameters

        :return: regularization loss
        """
        return 0.0

    def _get_norm_class_balance(self):
        return self.class_balance - torch.logsumexp(self.class_balance, dim=0)

    def _cll(self, votes, bid):
        '''Calculates class conditioned likelihood for batched votes.

        :param votes: current votes (batch)
        :param bid: batch id for current votes

        :return: 2-d torch tensor for class-conditioned likelihood for given votes and batch index.
        '''
        num_inst = votes.shape[0]

        za = self.accuracy.unsqueeze(2)
        za = torch.cat((za, -1 * za), dim=2)
        za = - torch.logsumexp(za, dim=2).unsqueeze(0).repeat(num_inst, 1, 1)

        z_plh = torch.zeros((self.num_df, 1)).to(self.device)
        zp = self.propensity.unsqueeze(1)
        zp = torch.cat((zp, z_plh), dim=1)
        zp = -torch.logsumexp(zp, dim=1).unsqueeze(0).unsqueeze(-1).repeat(num_inst, 1, self.num_classes)

        cp = self.propensity.unsqueeze(0).unsqueeze(-1).repeat(num_inst, 1, self.num_classes)
        ca = self.accuracy.unsqueeze(0).repeat(num_inst,1,1)
        ab = self.a[bid][:num_inst].to(self.device)
        cc = self.c[bid][:num_inst].to(self.device)
        cn = self.n[bid][:num_inst].to(self.device)

        cll = torch.sum(((ca*cc+cn+cp+za)*ab)+zp, dim=1)
        return cll


    def _create_minibatches(self, votes, batch_size, shuffle_rows=False):
        ''' Create (shuffled) batched votes for parallelized estimation

        :param votes: Full PLFs votes input.
        :param batch_size: # of instances in one batch
        :param shuffle_rows: Decides if rows of given votse need to shuffle

        :return:  3-d Numpy array for batched votes in shape [# batch, # instance, # plfs]
        '''
        if shuffle_rows:
            index = np.arange(np.shape(votes)[0])
            np.random.shuffle(index)
            votes = votes[index, :]

        batches = [
            torch.LongTensor(votes[i * batch_size: (i + 1) * batch_size, :].astype(np.int32))
            for i in range(int(np.ceil(votes.shape[0] / batch_size)))
        ]

        return batches


