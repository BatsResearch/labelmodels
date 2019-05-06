from labelmodels import HMM
import numpy as np
from scipy import sparse
import unittest


class TestHMM(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        pass

    def test_estimate_label_model_binary(self):
        n = 10
        k = 2

        accuracies = np.array([[.9, .8],
                               [.6, .7],
                               [.5, .5],
                               [.7, .6],
                               [.8, .8],
                               [.9, .8],
                               [.6, .7],
                               [.5, .5],
                               [.7, .6],
                               [.8, .8]])
        propensities = np.array([.6] * n)
        start_balance = np.array([.2, .8])
        transitions = np.array([[.5, .5], [.2, .8]])

        labels_train, seq_starts_train, gold_train = _generate_data(
            2000, 8, 12, n, accuracies, propensities, start_balance, transitions
        )

        model = HMM(k, n, acc_prior=0.0)
        model.estimate_label_model(labels_train, seq_starts_train)

        for i in range(n):
            for j in range(k):
                diff = accuracies[i, j] - model.get_accuracies()[i, j]
                self.assertAlmostEqual(diff, 0.0, places=1)
        for i in range(n):
            diff = propensities[i] - model.get_propensities()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for i in range(k):
            diff = start_balance[i] - model.get_start_balance()[i]
            self.assertAlmostEqual(diff, 0.0, places=1)
        for i in range(k):
            for j in range(k):
                diff = transitions[i, j] - model.get_transition_matrix()[i, j]
                self.assertAlmostEqual(diff, 0.0, places=1)

    def test_viterbi(self):
        n = 10
        k = 2

        accuracies = np.array([[.9, .8],
                               [.6, .7],
                               [.5, .5],
                               [.7, .6],
                               [.8, .8],
                               [.9, .8],
                               [.6, .7],
                               [.5, .5],
                               [.7, .6],
                               [.8, .8]])
        propensities = np.array([.6] * n)
        start_balance = np.array([.2, .8])
        transitions = np.array([[.5, .5], [.2, .8]])

        labels_train, seq_starts_train, gold_train = _generate_data(
            500, 8, 12, n, accuracies, propensities, start_balance, transitions
        )

        model = HMM(k, n, acc_prior=0.0)
        model.estimate_label_model(labels_train, seq_starts_train)

        for i in range(n):
            for j in range(k):
                diff = accuracies[i, j] - model.get_accuracies()[i, j]
                self.assertLess(np.abs(diff), 0.1)
        for i in range(n):
            diff = propensities[i] - model.get_propensities()[i]
            self.assertLess(np.abs(diff), 0.1)
        for i in range(k):
            diff = start_balance[i] - model.get_start_balance()[i]
            self.assertLess(np.abs(diff), 0.1)
        for i in range(k):
            for j in range(k):
                diff = transitions[i, j] - model.get_transition_matrix()[i, j]
                self.assertLess(np.abs(diff), 0.1)

        predictions = model.viterbi(labels_train, seq_starts_train)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == gold_train[i]:
                correct += 1
        accuracy = correct / float(len(predictions))
        self.assertGreaterEqual(accuracy, .80)


def _generate_data(num_seqs, min_seq, max_seq, num_lfs, accuracies,
                   propensities, start_balance, transitions):
    # Generates sequence starts
    seq_starts = np.zeros((num_seqs,), dtype=np.int)
    total_len = 0
    for i in range(num_seqs):
        seq_len = np.random.randint(min_seq, max_seq + 1)
        total_len += seq_len
        if i + 1 < num_seqs:
            seq_starts[i + 1] = total_len

    # Generates sequences of gold labels
    gold = np.zeros((total_len,), dtype=np.int)
    next_start = 0
    for i in range(total_len):
        if next_start < len(seq_starts) and i == seq_starts[next_start]:
            balance = start_balance
            next_start += 1
        else:
            balance = np.squeeze(transitions[gold[i - 1] - 1])

        gold[i] = np.argmax(np.random.multinomial(1, balance)) + 1

    # Generates labeling function outputs conditioned on gold labels
    row = []
    col = []
    val = []
    for i in range(total_len):
        for j in range(num_lfs):
            if np.random.random() < propensities[j]:
                row.append(i)
                col.append(j)
                if np.random.random() < accuracies[j, gold[i] - 1]:
                    val.append(gold[i])
                else:
                    p_mistake = 1 / (len(start_balance) - 1)
                    dist = [p_mistake] * (len(start_balance) + 1)
                    dist[0] = 0
                    dist[gold[i]] = 0
                    val.append(np.argmax(np.random.multinomial(1, dist)))

    labels = sparse.coo_matrix((val, (row, col)), shape=(total_len, num_lfs))

    return labels, seq_starts, gold


if __name__ == '__main__':
    unittest.main()
