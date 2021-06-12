from labelmodels import NaiveBayes
import numpy as np
from scipy import sparse
import test.util as util
import torch
import unittest


class TestNaiveBayes(unittest.TestCase):

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

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = NaiveBayes(2, n, acc_prior=0.0, balance_prior=0.0)
        model.estimate_label_model(labels_train)

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

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = NaiveBayes(3, n, acc_prior=0.0, balance_prior=0.0)
        model.estimate_label_model(labels_train)

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

    def test_get_most_probable_labels_binary(self):
        m = 10000
        n = 5
        k = 2

        model = NaiveBayes(k, n)
        with torch.no_grad():
            model.class_balance[0] = 0
            model.class_balance[1] = 0.5
            for i in range(n):
                model.propensity[i] = 2
                for j in range(k):
                    model.accuracy[i, j] = 2

        labels_train, gold_train = _generate_data(
            m, n,
            model.get_accuracies(),
            model.get_propensities(),
            model.get_class_balance())

        # Checks label inference
        labels = model.get_most_probable_labels(labels_train)
        correct = 0
        for i in range(m):
            if gold_train[i] == labels[i]:
                correct += 1

        self.assertGreater(float(correct) / m, .95)

    def test_get_most_probable_labels_multiclass(self):
        m = 10000
        n = 5
        k = 3

        model = NaiveBayes(k, n)
        with torch.no_grad():
            model.class_balance[0] = 0
            model.class_balance[1] = 0.5
            model.class_balance[2] = 0.5
            for i in range(n):
                model.propensity[i] = 2
                for j in range(k):
                    model.accuracy[i, j] = 2

        labels_train, gold_train = _generate_data(
            m, n,
            model.get_accuracies(),
            model.get_propensities(),
            model.get_class_balance())

        # Checks label inference
        labels = model.get_most_probable_labels(labels_train)
        correct = 0
        for i in range(m):
            if gold_train[i] == labels[i]:
                correct += 1

        self.assertGreater(float(correct) / m, .95)

    def test_estimate_model_input_formats(self):
        m = 1000
        n = 3

        accuracies = np.array([[.8, .8, .8],
                               [.8, .8, .8],
                               [.8, .8, .8]])
        propensities = np.array([.5] * n)
        class_balance = np.array([.1, .1, .8])

        labels_train, _ = _generate_data(
            m, n, accuracies, propensities, class_balance)

        # Trains the model on the generated data
        model = NaiveBayes(3, n)
        model.estimate_label_model(labels_train)
        accuracies = model.get_accuracies()
        propensities = model.get_propensities()
        class_balance = model.get_class_balance()

        # Checks that other input formats work and do not change the results
        for data in util.get_all_formats(labels_train):
            model = NaiveBayes(3, n)
            model.estimate_label_model(data)
            diff = np.sum(np.abs(accuracies - model.get_accuracies()))
            self.assertAlmostEqual(float(diff), 0.0)
            diff = np.sum(np.abs(propensities - model.get_propensities()))
            self.assertAlmostEqual(float(diff), 0.0)
            diff = np.sum(np.abs(class_balance - model.get_class_balance()))
            self.assertAlmostEqual(float(diff), 0.0)

    def test_get_label_input_formats(self):
        m = 1000
        n = 3

        accuracies = np.array([[.8, .8, .8],
                               [.8, .8, .8],
                               [.8, .8, .8]])
        propensities = np.array([.5] * n)
        class_balance = np.array([.1, .1, .8])

        labels_train, _ = _generate_data(
            m, n, accuracies, propensities, class_balance)

        # Gets the label distribution for the generated data
        model = NaiveBayes(3, n, init_acc=0.8)
        distribution = model.get_label_distribution(labels_train)

        # Checks that other input formats work and do not change the results
        for data in util.get_all_formats(labels_train):
            model = NaiveBayes(3, n, init_acc=0.8)
            new_distribution = model.get_label_distribution(data)
            diff = np.sum(np.abs(distribution - new_distribution))
            self.assertAlmostEqual(float(diff), 0.0)


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
