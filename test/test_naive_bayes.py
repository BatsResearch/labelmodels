from labelmodels import naive_bayes
import numpy as np
from scipy import sparse
import unittest


class TestNaiveBayes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_estimate_label_model_binary(self):
        m = 10000
        n = 10
        accuracies = np.array([.9, .9, .9, .8, .8, .8, .7, .7, .7, .4])
        propensities = np.array([.15] * n)
        class_balance = np.array([.1, .9])

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = naive_bayes.NaiveBayes(2, n, learn_class_balance=True,
                                       acc_prior=0.0)
        model.estimate_label_model(labels_train)

        for i in range(n):
            diff = accuracies[i] - model.get_accuracies()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for i in range(n):
            diff = propensities[i] - model.get_propensities()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for i in range(len(class_balance)):
            diff = class_balance[i] - model.get_class_balance()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_estimate_label_model_multiclass(self):
        m = 10000
        n = 10
        accuracies = np.array([.9, .9, .9, .8, .8, .8, .7, .7, .7, .4])
        propensities = np.array([.15] * n)
        class_balance = np.array([.1, .1, .2, .2, .4])

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = naive_bayes.NaiveBayes(5, n, learn_class_balance=True,
                                       acc_prior=0.0)
        model.estimate_label_model(labels_train)

        for i in range(n):
            diff = accuracies[i] - model.get_accuracies()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for i in range(n):
            diff = propensities[i] - model.get_propensities()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for i in range(len(class_balance)):
            diff = class_balance[i] - model.get_class_balance()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_estimate_label_model_strong_prior(self):
        m = 10000
        n = 10
        accuracies = np.array([.9, .9, .9, .8, .8, .8, .7, .7, .7, .4])
        propensities = np.array([.15] * n)
        class_balance = np.array([.1, .9])

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = naive_bayes.NaiveBayes(2, n, init_lf_acc=.6, acc_prior=0.5)
        model.estimate_label_model(labels_train)

        for i in range(n):
            diff = 0.6 - model.get_accuracies()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for i in range(n):
            diff = propensities[i] - model.get_propensities()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_get_label_distribution_binary(self):
        m = 10000
        n = 10

        accuracies = np.array([.8, .8, .8, .8, .8, .8, .8, .8, .8, .8])
        propensities = np.array([.5] * n)
        class_balance = np.array([.1, .9])

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = naive_bayes.NaiveBayes(2, n, init_lf_acc=.8)
        model.class_balance[0] = -1.1
        model.class_balance[1] = 1.1

        # Checks that accuracies and class balance were initialized correctly
        diff = np.sum(np.abs(accuracies - model.get_accuracies()))
        self.assertAlmostEqual(diff, 0.0, places=3)
        diff = np.sum(np.abs(class_balance - model.get_class_balance()))
        self.assertAlmostEqual(diff, 0.0, places=3)

        # Checks label inference
        labels = model.get_label_distribution(labels_train)
        correct = 0
        for i in range(m):
            if gold_train[i] == np.argmax(labels[i, :]) + 1:
                correct += 1

        self.assertGreater(float(correct) / m, .925)

    def test_get_label_distribution_multiclass(self):
        m = 10000
        n = 10

        accuracies = np.array([.8, .8, .8, .8, .8, .8, .8, .8, .8, .8])
        propensities = np.array([.5] * n)
        class_balance = np.array([.1, .1, .8])

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = naive_bayes.NaiveBayes(3, n, init_lf_acc=.8)
        model.class_balance[0] = -1.08
        model.class_balance[1] = -1.08
        model.class_balance[2] = 1

        # First checks that accuracies were initialized correctly
        diff = np.sum(np.abs(accuracies - model.get_accuracies()))
        self.assertAlmostEqual(diff, 0.0, places=3)
        diff = np.sum(np.abs(class_balance - model.get_class_balance()))
        self.assertAlmostEqual(diff, 0.0, places=3)

        # Checks label inference
        labels = model.get_label_distribution(labels_train)
        correct = 0
        for i in range(m):
            if gold_train[i] == np.argmax(labels[i, :]) + 1:
                correct += 1

        self.assertGreater(float(correct) / m, .85)


def _generate_data(m, n, accuracies, propensities, class_balance):
    gold = np.zeros((m,), dtype=np.int16)
    row = []
    col = []
    val = []

    for i in range(m):
        gold[i] = np.argmax(np.random.multinomial(1, class_balance)) + 1
        for j in range(n):
            if np.random.random() < propensities[j]:
                row.append(i)
                col.append(j)
                if np.random.random() < accuracies[j]:
                    val.append(gold[i])
                else:
                    p_mistake = 1 / (len(class_balance) - 1)
                    dist = [p_mistake] * (len(class_balance) + 1)
                    dist[0] = 0
                    dist[gold[i]] = 0
                    val.append(np.argmax(np.random.multinomial(1, dist)))

    labels = sparse.coo_matrix((val, (row, col)), shape=(m, n)).tocsr()
    return labels, gold


if __name__ == '__main__':
    unittest.main()
