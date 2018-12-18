import logging
import numpy as np
import torch
import torch.nn as nn


class LabelModel(nn.Module):
    """Parent class for all generative label models.

    Subclasses should implement at least forward(), estimate_label_model(), and
    get_labels().
    """

    def forward(self, *args):
        """Computes the marginal log-likelihood of a batch of observed labeling
        function outputs provided as input.

        :param args: batch of observed labeling function outputs
        :return: 1-d tensor of log-likelihoods, one for each input example
        """
        raise NotImplementedError

    def estimate_label_model(self, *args, config=None):
        """Learns the parameters of the label model from observed labeling
        function outputs.

        Subclasses that implement this method should call _do_estimate_label_model()
        if possible, to provide consistent behavior.

        :param args: observed labeling function outputs and related metadata
        :param config: an instance of LearningConfig. If none, will initialize
                       with default constructor
        """
        raise NotImplementedError

    def get_labels(self, *args):
        """

        :param args:
        :return:
        """
        raise NotImplementedError

    def _do_estimate_label_model(self, input_batcher, config):
        """Internal method for optimizing model parameters.

        :param input_batcher: generator that produces batches of inputs to
                              forward(). If forward() takes multiple arguments,
                              produces tuples.
        :param config: an instance of LearningConfig
        """
        init_random(config.random_seed)

        # Sets up optimization hyperparameters
        optimizer = torch.optim.SGD(
            self.parameters(), lr=config.step_size, momentum=config.momentum,
            weight_decay=0)
        if config.step_schedule is not None and config.step_size_mult is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.step_schedule, gamma=config.step_size_mult)
        else:
            scheduler = None

        # Iterates over epochs
        for epoch in range(config.epochs):
            logging.info('Epoch {}/{}'.format(epoch + 1, config.epochs))
            if scheduler is not None:
                scheduler.step()

            # Sets model to training mode
            self.train()
            running_loss = 0.0

            # Iterates over training data
            i_batch = 0
            for i_batch, inputs in enumerate(input_batcher):
                optimizer.zero_grad()
                log_likelihood = self(inputs)
                loss = -1 * torch.mean(log_likelihood)
                loss += self._get_regularization_loss()
                loss.backward()
                optimizer.step()

                running_loss += loss

            epoch_loss = running_loss / (i_batch + 1)
            logging.info('Train Loss: %.6f', epoch_loss)

            # if val is not None:
            #     # Sets model to validation mode
            #     self.eval()
            #     running_loss = 0.0
            #     running_corrects = 0
            #
            #     # Iterates over validation data
            #     for inputs, labels in val:
            #
            #         with torch.set_grad_enabled(False):
            #             outputs = model(inputs)
            #             loss = criterion(outputs, labels)
            #
            #             _, preds = torch.max(outputs, 1)
            #
            #         running_loss += loss.item() * inputs.size(0)
            #         running_corrects += torch.sum(preds == labels.data)
            #
            #     epoch_loss = running_loss / len(val.dataset)
            #     epoch_acc = running_corrects.double() / len(val.dataset)
            #     logging.info('Val.  Loss: %.4f Acc: %.4f', epoch_loss, epoch_acc)

    def _get_regularization_loss(self):
        return 0.0


class LearningConfig(object):
    """Container for hyperparameters used by label models during learning"""

    def __init__(self):
        """Initializes all hyperparameters to default values"""
        self.epochs = 5
        self.batch_size = 64
        self.step_size = 0.1
        self.step_schedule = None
        self.step_size_mult = None
        self.momentum = 0.0
        self.random_seed = 0


def init_random(seed):
    """Initializes PyTorch and NumPy random seeds.

    Also sets the CuDNN back end to deterministic.

    :param seed: integer to use as random seed
    """
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logging.info("Random seed: %d", seed)
