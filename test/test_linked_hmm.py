from labelmodels import LinkedHMM
import numpy as np
from scipy import sparse
import unittest


class TestLinkedHMM(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        pass

    def test_estimate_label_model_binary(self):
        n = 5
        k = 2

        accuracies = np.array([[.9, .8],
                               [.6, .7],
                               [.6, .6],
                               [.7, .6],
                               [.8, .8]])
        propensities = np.array([.9] * n)
        start_balance = np.array([.3, .7])
        transitions = np.array([[.5, .5], [.3, .7]])

        labels_train, seq_starts_train, gold_train = _generate_data(
            1000, 8, 12, n, accuracies, propensities, start_balance, transitions
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

    def test_estimate_label_model_multiclass(self):
        n = 5
        k = 3

        accuracies = np.array([[.9, .8, .9],
                               [.6, .7, .9],
                               [.6, .6, .9],
                               [.7, .6, .9],
                               [.8, .8, .9]])
        propensities = np.array([.9] * n)
        start_balance = np.array([.3, .3, .4])
        transitions = np.array([[.5, .3, .2],
                                [.3, .4, .3],
                                [.2, .5, .3]])

        labels_train, seq_starts_train, gold_train = _generate_data(
            1000, 8, 12, n, accuracies, propensities, start_balance, transitions
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
        m = 500
        n = 10
        k = 3

        model = HMM(k, n, acc_prior=0.0)

        model.start_balance[0] = 0
        model.start_balance[1] = 0.5
        for i in range(n):
            model.propensity[i] = 2
            for j in range(k):
                model.accuracy[i, j] = 2
        for i in range(k):
            for j in range(k):
                model.transitions[i, j] = 1 if i == j else 0

        labels_train, seq_starts_train, gold_train = _generate_data(
            m, 8, 12, n,
            model.get_accuracies(),
            model.get_propensities(),
            model.get_start_balance(),
            model.get_transition_matrix())

        predictions = model.viterbi(labels_train, seq_starts_train)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == gold_train[i]:
                correct += 1
        accuracy = correct / float(len(predictions))
        self.assertGreaterEqual(accuracy, .95)

    def test_get_label_distribution(self):
        m = 500
        n1 = 3
        n2 = 5
        k = 3

        model = LinkedHMM(k, n1, n2, acc_prior=0.0)

        model.start_balance[0] = 0
        model.start_balance[1] = 0.5
        for i in range(n1):
            model.propensity[i] = 0
            for j in range(k):
                model.accuracy[i, j] = 1
        for i in range(n2):
            model.link_propensity[i] = 0
            model.link_accuracy[i] = 1
        for i in range(k):
            for j in range(k):
                model.transitions[i, j] = 1 if i == j else 0

        labels, links, seq_starts, gold = _generate_data(
            m, 8, 12, n1, n2,
            model.get_label_accuracies(),
            model.get_link_accuracies(),
            model.get_label_propensities(),
            model.get_link_propensities(),
            model.get_start_balance(),
            model.get_transition_matrix())

        p_unary, p_pairwise = model.get_label_distribution(
            labels, links, seq_starts)

        # Makes predictions using both unary and pairwise marginals
        pred_unary = np.argmax(p_unary, axis=1) + 1
        pred_pairwise = np.zeros((labels.shape[0],), dtype=np.int)
        next_seq = 0
        for i in range(labels.shape[0] - 1):
            if next_seq == len(seq_starts) or i < seq_starts[next_seq] - 1:
                # i is neither the start nor end of a sequence
                pred_pairwise[i+1] = np.argmax(p_pairwise[i][pred_pairwise[i]])
            elif i == seq_starts[next_seq]:
                # i is the start of a sequence
                a, b = np.unravel_index(p_pairwise[i].argmax(), (k, k))
                pred_pairwise[i], pred_pairwise[i + 1] = a, b
                next_seq += 1
            else:
                # i is the end of a sequence
                pass
        pred_pairwise += 1

        # Checks that predictions are accurate
        for predictions in (pred_unary,):
        #for predictions in (pred_unary, pred_pairwise):
            correct = 0
            for i in range(len(predictions)):
                if predictions[i] == gold[i]:
                    correct += 1
            accuracy = correct / float(len(predictions))
            self.assertGreaterEqual(accuracy, .95)


def _generate_data(num_seqs, min_seq, max_seq, num_label_funcs, num_link_funcs,
                   label_accs, link_accs, label_propensities, link_propensities,
                   start_balance, transitions):
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
        for j in range(num_label_funcs):
            if np.random.random() < label_propensities[j]:
                row.append(i)
                col.append(j)
                if np.random.random() < label_accs[j, gold[i] - 1]:
                    val.append(gold[i])
                else:
                    p_mistake = 1 / (len(start_balance) - 1)
                    dist = [p_mistake] * (len(start_balance) + 1)
                    dist[0] = 0
                    dist[gold[i]] = 0
                    val.append(np.argmax(np.random.multinomial(1, dist)))

    labels = sparse.coo_matrix((val, (row, col)), shape=(total_len, num_label_funcs))

    # Generates linking function outputs conditioned on gold labels
    row = []
    col = []
    val = []
    next_seq = 0
    for i in range(total_len):
        if next_seq < len(seq_starts) and i == seq_starts[next_seq]:
            next_seq += 1
        else:
            for j in range(num_link_funcs):
                if np.random.random() < link_propensities[j]:
                    row.append(i)
                    col.append(j)
                    if np.random.random() < link_accs[j]:
                        val.append(1 if gold[i-1] == gold[i] else -1)
                    else:
                        val.append(-1 if gold[i-1] == gold[i] else 1)

    links = sparse.coo_matrix((val, (row, col)), shape=(total_len, num_link_funcs))

    return labels, links, seq_starts, gold


if __name__ == '__main__':
    unittest.main()
