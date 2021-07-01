from labelmodels import NPLM
import numpy as np
from scipy import sparse
import test.util as util
import torch
import unittest
from random import sample
from copy import deepcopy as dc


class Experiment:
    def __init__(self, name, num_classes, fid2clusters, lm_annot_votes, lm_train_votes, lm_annot_labels=None):
        self.name = name
        self.fid2clusters = fid2clusters
        self.lm_annot_votes = lm_annot_votes
        self.lm_train_votes = lm_train_votes
        self.lm_annot_labels = lm_annot_labels
        self.num_classes = num_classes
        self.num_df = len(fid2clusters)

    def set_soft_labels(self, soft_labels):
        pass


class LMTask:
    def __init__(self, name, num_classes, fid2clusters, preset_cb=None, device='cuda:0'):
        self.name = name

        self.labelmodel = NPLM(num_classes=num_classes,
                               fid2clusters=fid2clusters,
                               preset_cb=preset_cb,
                               device=device)

    def annotate(self, lm_annot_votes, lm_train_votes=None):
        if lm_train_votes is not None:
            self.labelmodel.estimate_label_model(lm_train_votes)
        return self.labelmodel.estimate_label_model(lm_annot_votes)

    def get_accuracy(self):
        return self.labelmodel.get_accuracies()

    def get_propensity(self):
        return self.labelmodel.get_propensities()

    def get_class_balance(self):
        return self.labelmodel.get_class_balance()


def Workflow(experimental_data):
    lm_task = LMTask('test', num_classes=experimental_data.num_classes,
                     fid2clusters=experimental_data.fid2clusters,
                     device='cpu')

    lm_annot_soft_labels = lm_task.annotate(lm_annot_votes=experimental_data.lm_annot_votes,
                                            lm_train_votes=experimental_data.lm_train_votes)
    output = []
    output.append(lm_task.get_accuracy())
    output.append(lm_task.get_class_balance())
    output.append(lm_task.get_propensity())
    return output


def setup():
    simple_fid2clusters = {
        0: [[1], [2, 3]],
        1: [[2], [1, 3]],
        2: [[3], [1, 2]]
    }
    num_sources = len(simple_fid2clusters)
    num_classes = 4
    num_annot_inst = 4096 * 16

    labelmodel_annotation_votes = np.random.randint(2, size=(num_annot_inst, num_sources))
    labelmodel_training_votes = labelmodel_annotation_votes
    labelmodel_annotation_labels = np.random.randint(num_classes, size=(num_annot_inst, 1)) + 1
    test_data = Experiment('simple-tests',
                           num_classes=num_classes,
                           fid2clusters=simple_fid2clusters,
                           lm_annot_votes=labelmodel_annotation_votes,
                           lm_train_votes=labelmodel_training_votes,
                           lm_annot_labels=labelmodel_annotation_labels)

    return test_data


def setup_test(fid2clusters, accuracies, class_balance, m=4096*8, abstention=None):
    votes, gold = _generate_data(m, fid2clusters, accuracies, class_balance, abstention=abstention)

    return votes, gold


def close_estimation(model_acc, true_acc, thresh=0.05, verbose=True):
    # assert model_acc.shape == true_acc.shape
    res = torch.allclose(torch.Tensor(model_acc), torch.Tensor(true_acc), atol=thresh)
    if verbose:
        print(res)
    return res


def actual_cb(gold):
    unique, counts = np.unique(gold, return_counts=True)
    return counts / sum(counts)


class TestNPLM(unittest.TestCase):
    def test_general_accuracy_recovery_0(self):
        print('Testing Accuracy Recovery Rate for GenLM - 0')
        true_cb_0 = [1 / 3, 1 / 3, 1 / 3]
        true_acc_0 = np.array(
            [[.8, .7, .6],
             [.75, .7, .7],
             [.5, .7, .65],
             [.8, .8, .75],
             [.9, .7, .8]])
        fid2clusters = {
            0: [[1], [2, 3]],
            1: [[1], [2, 3]],
            2: [[1, 2], [3]],
            3: [[1, 2], [3]],
            4: [[1, 3], [2]]
        }
        votes, gold = setup_test(fid2clusters, true_acc_0, true_cb_0)
        test_data_0 = Experiment('acc-tests-0',
                                 num_classes=3,
                                 fid2clusters=fid2clusters,
                                 lm_annot_votes=votes,
                                 lm_train_votes=votes,
                                 lm_annot_labels=gold)

        acc_0, cb_0, _ = Workflow(experimental_data=test_data_0)

        #print(acc_0 - true_acc_0)
        self.assertTrue(close_estimation(acc_0, true_acc_0))
        self.assertTrue(close_estimation(cb_0, true_cb_0))

    def test_general_accuracy_recovery_1(self):
        print('Testing Accuracy Recovery Rate for GenLM - 1')
        true_cb_1 = [.5, .3, .2]
        true_acc_1 = np.array(
            [[.8, .7, .6],
             [.8, .7, .6],
             [.5, .9, .6],
             [.8, .7, .6],
             [.9, .7, .6]])
        fid2clusters = {
            0: [[1], [2], [3]],
            1: [[1], [2, 3]],
            2: [[1, 2], [3]],
            3: [[1], [2], [3]],
            4: [[1], [2], [3]]
        }
        votes, gold = setup_test(fid2clusters, true_acc_1, true_cb_1)
        test_data_1 = Experiment('acc-tests-1',
                                 num_classes=3,
                                 fid2clusters=fid2clusters,
                                 lm_annot_votes=votes,
                                 lm_train_votes=votes,
                                 lm_annot_labels=gold)

        acc_1, cb_1, _ = Workflow(experimental_data=test_data_1)

        self.assertTrue(close_estimation(acc_1, true_acc_1))
        self.assertTrue(close_estimation(cb_1, true_cb_1))

    def test_general_accuracy_recovery_2(self):
        print('Testing Accuracy Recovery Rate for GenLM - 2')
        true_cb_2 = [.4, .2, .4]
        true_acc_2 = np.array(
            [[.8, .7, .6],
             [.8, .8, .7],
             [.5, .7, .9],
             [.8, .6, .7],
             [.9, .7, .6],
             [.8, .5, .6]])
        fid2clusters = {
            0: [[1], [2], [3]],
            1: [[2], [1, 3]],
            2: [[1, 2], [3]],
            3: [[1], [2], [3]],
            4: [[1], [2], [3]],
            5: [[3, 2], [1]]
        }
        votes, gold = setup_test(fid2clusters, true_acc_2, true_cb_2)

        test_data_2 = Experiment('acc-tests-2',
                                 num_classes=3,
                                 fid2clusters=fid2clusters,
                                 lm_annot_votes=votes,
                                 lm_train_votes=votes,
                                 lm_annot_labels=gold)

        acc_2, cb_2, _ = Workflow(experimental_data=test_data_2)
        print(acc_2 - true_acc_2)
        print(true_cb_2 - cb_2)
        self.assertTrue(close_estimation(acc_2, true_acc_2))
        self.assertTrue(close_estimation(cb_2, true_cb_2))

    def test_general_accuracy_recovery_3(self):
        print('Testing Accuracy Recovery Rate for GenLM - 3 with Abstention')
        true_cb_3 = [.4, .2, .4]
        true_acc_3 = np.array(
            [[.8, .7, .6],
             [.8, .7, .7],
             [.5, .7, .9],
             [.8, .6, .7],
             [.9, .8, .6],
             [.8, .5, .8]])
        fid2clusters = {
            0: [[1], [2], [3]],
            1: [[2], [1, 3]],
            2: [[1, 2], [3]],
            3: [[1], [2], [3]],
            4: [[1], [2, 3]],
            5: [[3, 2], [1]]
        }

        abstention = [0.8, 0.9, 0.8, 0.7, 0.8, 0.9]
        votes, gold = setup_test(fid2clusters, true_acc_3, true_cb_3,
                                 abstention=abstention)

        tv = dc(votes)
        tv[tv == -1] = 100
        tv[tv < 100] = 0
        tv = tv / 100
        est_abst = np.mean(tv, axis=0)
        # print(est_abst)

        test_data_3 = Experiment('acc-tests-2',
                                 num_classes=3,
                                 fid2clusters=fid2clusters,
                                 lm_annot_votes=votes,
                                 lm_train_votes=votes,
                                 lm_annot_labels=gold)
        acc_3, cb_3, prp_3 = Workflow(test_data_3)

        self.assertTrue(close_estimation(acc_3, true_acc_3))
        self.assertTrue(close_estimation(cb_3, true_cb_3))
        self.assertTrue(close_estimation(prp_3, abstention))

    def test_general_accuracy_recovery_4(self):
        print('Testing Accuracy Recovery Rate for GenLM - 4 with Abstention')
        true_cb_3 = [.4, .2, .4]
        true_acc_3 = np.array(
            [[.8, .7, .6],
             [.8, .6, .7],
             [.5, .7, .9],
             [.8, .6, .7],
             [.9, .8, .6],
             [.8, .5, .8]])
        fid2clusters = {
            0: [[1], [2], [3]],
            1: [[2], [1, 3]],
            2: [[1, 2], [3]],
            3: [[1], [2], [3]],
            4: [[1], [2, 3]],
            5: [[3, 2], [1]]
        }

        abstention = [0.8, 0.9, 0.8, 0.7, 0.8, 0.9]
        votes, gold = setup_test(fid2clusters, true_acc_3, true_cb_3,
                                 abstention=abstention)

        tv = dc(votes)
        tv[tv == -1] = 100
        tv[tv < 100] = 0
        tv = tv / 100
        est_abst = np.mean(tv, axis=0)
        print(est_abst)

        test_data_3 = Experiment('acc-tests-4',
                                 num_classes=3,
                                 fid2clusters=fid2clusters,
                                 lm_annot_votes=votes,
                                 lm_train_votes=votes,
                                 lm_annot_labels=gold)
        acc_3, cb_3, prp_3 = Workflow(test_data_3)
        self.assertTrue(close_estimation(acc_3, true_acc_3))
        self.assertTrue(close_estimation(cb_3, true_cb_3))
        self.assertTrue(close_estimation(prp_3, abstention))


def _generate_data(m, fid2clusters, accuracies, class_balance, abstention=None):
    """
    Generate synthetic data

    :param m: number of examples
    :param n: number of sources
    :param fid2clusters: feature id clustering
    :param accuracies: n x k matrix of accuracies, where k is number of classes
    :param class_balance: k-dim vector representing prior over classes
    :param abstention: n-dim vector representing prob of not abstention
    :return: m x n matrix of features, m-dim vector of gold class labels
    """
    n = len(fid2clusters)
    gold = np.zeros((m,), dtype=np.int16)
    votes = np.zeros((m, n), dtype=np.int16)

    for i in range(m):
        k = np.argmax(np.random.multinomial(1, class_balance))
        gold[i] = k + 1
        for j in range(n):
            # Collects correct and incorrect clusters
            correct = []
            incorrect = []
            for cid, cluster in enumerate(fid2clusters[j]):
                if k + 1 in cluster:
                    correct.append(cid)
                else:
                    incorrect.append(cid)
            if np.random.random() < accuracies[j, k]:
                votes[i, j] = np.random.choice(correct)
            else:
                votes[i, j] = np.random.choice(incorrect)

    if abstention is not None:
        for idx, prob in enumerate(abstention):
            votes[np.array(sample(range(m), int((1 - prob) * m))), idx] = -1

    return votes, gold


if __name__ == '__main__':
    unittest.main()
