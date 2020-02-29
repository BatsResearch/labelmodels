from labelmodels import SemiSupervisedNaiveBayes
import numpy as np
from scipy import sparse
import unittest


class TestSemiSupervisedNaiveBayes(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        pass

    def test_estimate_label_model_binary(self):
        m = 25000
        n = 5
        accuracies = np.array([[.9, .8],
                               [.6, .7],
                               [.5, .5],
                               [.7, .6],
                               [.8, .8]])
        propensities = np.array([.3] * n)
        class_balance = np.array([.4, .6])
        obs_fraction = .1

        votes_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        labels_train = np.array(gold_train, copy=True)
        idx = np.arange(len(gold_train))
        masked = np.random.choice(idx, size=int(len(idx) * (1 - obs_fraction)), replace=False)
        labels_train[masked] = 0

        model = SemiSupervisedNaiveBayes(2, n)
        model.estimate_label_model(votes_train, labels_train)

        for j in range(n):
            for k in range(2):
                diff = accuracies[j, k] - model.get_accuracies()[j, k]
                self.assertAlmostEqual(diff, 0.0, places=1)
        for j in range(n):
            diff = propensities[j] - model.get_propensities()[j]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for k in range(len(class_balance)):
            diff = class_balance[k] - model.get_class_balance()[k]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_estimate_label_model_binary_high_obs(self):
        m = 25000
        n = 5
        accuracies = np.array([[.9, .8],
                               [.6, .7],
                               [.5, .5],
                               [.7, .6],
                               [.8, .8]])
        propensities = np.array([.3] * n)
        class_balance = np.array([.4, .6])
        obs_fraction = .9

        votes_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        labels_train = np.array(gold_train, copy=True)
        idx = np.arange(len(gold_train))
        masked = np.random.choice(idx, size=int(len(idx) * (1 - obs_fraction)), replace=False)
        labels_train[masked] = 0

        model = SemiSupervisedNaiveBayes(2, n)
        model.estimate_label_model(votes_train, labels_train)

        for j in range(n):
            for k in range(2):
                diff = accuracies[j, k] - model.get_accuracies()[j, k]
                self.assertAlmostEqual(diff, 0.0, places=1)
        for j in range(n):
            diff = propensities[j] - model.get_propensities()[j]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for k in range(len(class_balance)):
            diff = class_balance[k] - model.get_class_balance()[k]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_estimate_label_model_multiclass(self):
        m = 25000
        n = 5
        accuracies = np.array([[.9, .8, .5],
                               [.6, .7, .5],
                               [.5, .5, .9],
                               [.7, .6, .7],
                               [.8, .8, .7]])
        propensities = np.array([.2] * n)
        class_balance = np.array([.3, .4, .3])
        obs_fraction = .1

        votes_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        labels_train = np.array(gold_train, copy=True)
        idx = np.arange(len(gold_train))
        masked = np.random.choice(idx, size=int(len(idx) * (1 - obs_fraction)), replace=False)
        labels_train[masked] = 0

        model = SemiSupervisedNaiveBayes(3, n)
        model.estimate_label_model(votes_train, labels_train)

        for j in range(n):
            for k in range(2):
                diff = accuracies[j, k] - model.get_accuracies()[j, k]
                self.assertAlmostEqual(diff, 0.0, places=1)
        for j in range(n):
            diff = propensities[j] - model.get_propensities()[j]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for k in range(len(class_balance)):
            diff = class_balance[k] - model.get_class_balance()[k]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_estimate_label_model_multiclass_high_obs(self):
        m = 25000
        n = 5
        accuracies = np.array([[.9, .8, .5],
                               [.6, .7, .5],
                               [.5, .5, .9],
                               [.7, .6, .7],
                               [.8, .8, .7]])
        propensities = np.array([.2] * n)
        class_balance = np.array([.3, .4, .3])
        obs_fraction = .9

        votes_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        labels_train = np.array(gold_train, copy=True)
        idx = np.arange(len(gold_train))
        masked = np.random.choice(idx, size=int(len(idx) * (1 - obs_fraction)), replace=False)
        labels_train[masked] = 0

        model = SemiSupervisedNaiveBayes(3, n)
        model.estimate_label_model(votes_train, labels_train)

        for j in range(n):
            for k in range(2):
                diff = accuracies[j, k] - model.get_accuracies()[j, k]
                self.assertAlmostEqual(diff, 0.0, places=1)
        for j in range(n):
            diff = propensities[j] - model.get_propensities()[j]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for k in range(len(class_balance)):
            diff = class_balance[k] - model.get_class_balance()[k]
            self.assertAlmostEqual(diff, 0.0, places=1)


def _generate_data(m, n, accuracies, propensities, class_balance):
    gold = np.zeros((m,), dtype=np.int16)
    row = []
    col = []
    val = []

    for i in range(m):
        k = np.argmax(np.random.multinomial(1, class_balance))
        gold[i] = k + 1
        for j in range(n):
            if np.random.random() < propensities[j]:
                row.append(i)
                col.append(j)
                if np.random.random() < accuracies[j, k]:
                    val.append(gold[i])
                else:
                    p_mistake = 1 / (len(class_balance) - 1)
                    dist = [p_mistake] * (len(class_balance) + 1)
                    dist[0] = 0
                    dist[gold[i]] = 0
                    val.append(np.argmax(np.random.multinomial(1, dist)))

    labels = sparse.coo_matrix((val, (row, col)), shape=(m, n))
    return labels, gold


if __name__ == '__main__':
    unittest.main()
