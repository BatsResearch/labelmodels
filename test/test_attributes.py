from labelmodels import AttributeLabelModel
import numpy as np
import unittest


class TestAttributeLabelModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        pass

    def test_estimate_label_model(self):
        m = 15000
        n = 4
        num_classes = 3
        class_balance = np.array([0.5, 0.25, 0.25])
        accuracies = np.array([[0.8, 0.7, 0.7],
                               [0.7, 0.8, 0.8],
                               [0.7, 0.6, 0.6],
                               [0.7, 0.6, 0.6]])
        specificities = np.array([[0.4, 0.2, 0.4],
                                  [0.3, 0.0, 0.7],
                                  [0.0, 0.3, 0.7],
                                  [0.2, 0.5, 0.3]])

        labels_train, gold_train = _generate_data(
            m, accuracies, specificities, class_balance)

        model = AttributeLabelModel(num_classes, n, acc_prior=0.0, balance_prior=0.0)
        model.estimate_label_model(labels_train)

        for j in range(n):
            for k in range(num_classes):
                diff = accuracies[j, k] - model.get_accuracies()[j, k]
                self.assertAlmostEqual(diff, 0.0, places=1)
        for j in range(n):
            for k in range(num_classes):
                diff = specificities[j, k] - model.get_specificities()[j, k]
                self.assertAlmostEqual(diff, 0.0, places=1)
        for k in range(num_classes):
            diff = class_balance[k] - model.get_class_balance()[k]
            self.assertAlmostEqual(diff, 0.0, places=1)

    def test_get_most_probable_labels(self):
        m = 5000
        n = 4
        num_classes = 3
        model = AttributeLabelModel(num_classes, n, acc_prior=0.0, balance_prior=0.0)

        model.class_balance[0] = 0
        model.class_balance[1] = 0.5
        model.class_balance[2] = 0.5

        for j in range(n):
            for k in range(num_classes):
                if k % 2 == 0:
                    model.accuracy[j, k] = 2
                    model.specificity[j, k] = 2
                else:
                    model.accuracy[j, k] = 1
                    model.specificity[j, k] = 1

        labels_train, gold_train = _generate_data(
            m,
            model.get_accuracies(),
            model.get_specificities(),
            model.get_class_balance())

        # Checks label inference
        labels = model.get_most_probable_labels(labels_train)
        correct = 0
        for i in range(m):
            if gold_train[i] == labels[i]:
                correct += 1

        self.assertGreater(float(correct) / m, .90)


def _generate_data(m, accuracies, specificities, class_balance):
    """
    Generates a 3D tensor of weak supervision source outputs
    along with associated ground truth labels

    Params
    ------
    m: number of points to generate
    class_balance: vector of length p where p is the number
                   classes, assumed to sum to 1
    accuracies: n x p matrix of class-conditional accuracies,
                where n is number of sources and p is number
                of classes
    specificities: n x p matrix of probabilities for number
                   of positive votes from each source, each
                   row is assumed to sum to 1
    Return
    ------
      - an m x n x p tensor with a 1 indicating that source j thinks data
        point m could have label k, and 0 otherwise
      - a vector of length m with the ground truth labels
    """
    n, p = accuracies.shape
    outputs = np.zeros((m, n, p), dtype=np.int16)
    labels = np.ndarray((m,), dtype=np.int16)

    for i in range(m):
        labels[i] = np.argmax(np.random.multinomial(1, class_balance))
        for j in range(n):
            # Generates the number of bits
            num_positive = np.argmax(np.random.multinomial(1, specificities[j])) + 1

            # Are all bits set? (source is abstaining)
            if num_positive == p:
                outputs[i, j, :] = 1
            else:
                # Is the correct bit set?
                if np.random.random() < accuracies[j, labels[i]]:
                    outputs[i, j, labels[i]] = 1
                    num_positive -= 1

                # Generates the other bits
                options = [i for i in range(p)]
                options.remove(labels[i])
                selected = np.random.choice(options, size=num_positive, replace=False)
                for k in selected:
                    outputs[i, j, k] = 1

    return outputs, labels


if __name__ == '__main__':
    unittest.main()
