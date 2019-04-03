from labelmodels import NaiveBayes
import numpy as np
import test.util as util
import unittest


class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        pass

    def test_estimate_label_model_binary(self):
        m = 10000
        class_balance = np.array([.1, .9])
        vote_dist = np.array([[[.5, .4, .1],
                               [.5, .1, .4]],
                              [[.2, .7, .1],
                               [.2, .1, .7]],
                              [[.3, .4, .3],
                               [.1, .3, .6]]])

        X, y = _generate_data(m, class_balance, vote_dist)
        n, k = vote_dist.shape[0], vote_dist.shape[1],

        model = NaiveBayes(k, n, entropy_prior=0.0)
        model.estimate_label_model(X)

        for i in range(len(class_balance)):
            diff = class_balance[i] - model.get_class_balance()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)

        learned_dist = model.get_vote_distribution()
        for j in range(n):
            for c in range(k):
                for l in range(k+1):
                    diff = vote_dist[l, c, j] - learned_dist[l, c, j]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_estimate_label_model_multiclass(self):
        m = 10000
        class_balance = np.array([.1, .1, .8])
        vote_dist = np.array([[[.4, .4, .1, .1],
                               [.4, .1, .4, .1],
                               [.4, .1, .1, .4]],
                              [[.4, .4, .1, .1],
                               [.4, .1, .4, .1],
                               [.4, .1, .1, .4]],
                              [[.4, .4, .1, .1],
                               [.4, .1, .4, .1],
                               [.4, .1, .1, .4]]])

        X, y = _generate_data(m, class_balance, vote_dist)
        n, k = vote_dist.shape[0], vote_dist.shape[1],

        model = NaiveBayes(k, n, learn_class_balance=True, entropy_prior=0.0)
        model.estimate_label_model(X)

        for i in range(len(class_balance)):
            diff = class_balance[i] - model.get_class_balance()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)

        learned_dist = model.get_vote_distribution()
        for j in range(n):
            for c in range(k):
                for l in range(k + 1):
                    diff = vote_dist[j, c, l] - learned_dist[j, c, l]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_estimate_label_model_strong_prior(self):
        m = 10000
        n = 10
        accuracies = np.array([.9, .9, .9, .8, .8, .8, .7, .7, .7, .4])
        propensities = np.array([.15] * n)
        class_balance = np.array([.1, .9])

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = NaiveBayes(2, n, init_lf_acc=.6, acc_prior=0.5)
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

        accuracies = np.array([.8] * n)
        propensities = np.array([.5] * n)
        class_balance = np.array([.1, .9])

        labels_train, gold_train = _generate_data(
            m, n, accuracies, propensities, class_balance)

        model = NaiveBayes(2, n, init_lf_acc=.8)
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
        class_balance = np.array([.1, .1, .8])
        vote_dist = np.array([[[.4, .4, .1, .1],
                               [.4, .1, .4, .1],
                               [.4, .1, .1, .4]],
                              [[.4, .4, .1, .1],
                               [.4, .1, .4, .1],
                               [.4, .1, .1, .4]],
                              [[.4, .4, .1, .1],
                               [.4, .1, .4, .1],
                               [.4, .1, .1, .4]]])

        X, y = _generate_data(m, class_balance, vote_dist)
        n, k = vote_dist.shape[0], vote_dist.shape[1],

        model = NaiveBayes(k, n, learn_class_balance=True, acc_prior=0.0)
        model.estimate_label_model(X)

        for i in range(len(class_balance)):
            diff = class_balance[i] - model.get_class_balance()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)

        learned_dist = model.get_vote_distribution()
        for j in range(n):
            for c in range(k):
                for l in range(k + 1):
                    diff = vote_dist[j, c, l] - learned_dist[j, c, l]
            self.assertAlmostEqual(diff, 0.0, places=1)

        # Checks label inference
        labels = model.get_label_distribution(X)
        correct = 0
        for i in range(m):
            if y[i] == np.argmax(labels[i, :]) + 1:
                correct += 1

        self.assertGreater(float(correct) / m, .8)

    def test_estimate_model_input_formats(self):
        m = 1000
        n = 3

        accuracies = np.array([.8] * 3)
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
            self.assertAlmostEqual(diff, 0.0)
            diff = np.sum(np.abs(propensities - model.get_propensities()))
            self.assertAlmostEqual(diff, 0.0)
            diff = np.sum(np.abs(class_balance - model.get_class_balance()))
            self.assertAlmostEqual(diff, 0.0)

    def test_get_label_input_formats(self):
        m = 1000
        n = 3

        accuracies = np.array([.8] * 3)
        propensities = np.array([.5] * n)
        class_balance = np.array([.1, .1, .8])

        labels_train, _ = _generate_data(
            m, n, accuracies, propensities, class_balance)

        # Gets the label distribution for the generated data
        model = NaiveBayes(3, n, init_lf_acc=0.8)
        distribution = model.get_label_distribution(labels_train)

        # Checks that other input formats work and do not change the results
        for data in util.get_all_formats(labels_train):
            model = NaiveBayes(3, n, init_lf_acc=0.8)
            new_distribution = model.get_label_distribution(data)
            diff = np.sum(np.abs(distribution - new_distribution))
            self.assertAlmostEqual(diff, 0.0)


def _generate_data(m, class_balance, vote_dist):
    n = vote_dist.shape[0]
    X = np.zeros((m, n), dtype=np.int16)
    y = np.zeros((m,), dtype=np.int16)

    for i in range(m):
        y[i] = np.argmax(np.random.multinomial(1, class_balance)) + 1
        for j in range(n):
            sample = np.random.multinomial(1, vote_dist[j, y[i] - 1, :])
            X[i, j] = np.argmax(sample)

    return X, y


if __name__ == '__main__':
    unittest.main()
