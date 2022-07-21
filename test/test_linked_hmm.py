import sys
mypath = "/Users/zhengxinyong/Desktop/labelmodels"
sys.path.append(mypath)

from labelmodels import LinkedHMM, LearningConfig
import numpy as np
from scipy import sparse
import torch
import unittest

class TestLinkedHMM(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def tearDown(self):
        pass

    # def test_estimate_label_model_binary(self):
    #     n1 = 5
    #     n2 = 3
    #     k = 2

    #     label_accuracies = np.array([[.9, .8],
    #                                  [.6, .7],
    #                                  [.6, .6],
    #                                  [.7, .6],
    #                                  [.8, .8]])
    #     link_accuracies = np.array([.8, .6, .8])
    #     label_propensities = np.array([.9] * n1)
    #     link_propensities = np.array([.9] * n1)
    #     start_balance = np.array([.3, .7])
    #     transitions = np.array([[.5, .5], [.3, .7]])

    #     labels, links, seq_starts, gold = _generate_data(
    #         1000, 8, 12, n1, n2,
    #         label_accuracies,
    #         link_accuracies,
    #         label_propensities,
    #         link_propensities,
    #         start_balance,
    #         transitions
    #     )

    #     model = LinkedHMM(k, n1, n2, acc_prior=0.0, balance_prior=0.0)
    #     config = LearningConfig()
    #     config.epochs = 3
    #     model.estimate_label_model(labels, links, seq_starts, config=config)

    #     for i in range(n1):
    #         for j in range(k):
    #             diff = label_accuracies[i, j] - model.get_accuracies()[i, j]
    #             self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(n2):
    #         for j in range(k):
    #             diff = link_accuracies[i] - model.get_link_accuracies()[i]
    #             self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(n1):
    #         diff = label_propensities[i] - model.get_propensities()[i]
    #         self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(n2):
    #         diff = link_propensities[i] - model.get_link_propensities()[i]
    #         self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(k):
    #         diff = start_balance[i] - model.get_start_balance()[i]
    #         self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(k):
    #         for j in range(k):
    #             diff = transitions[i, j] - model.get_transition_matrix()[i, j]
    #             self.assertAlmostEqual(diff, 0.0, places=1)

    # def test_estimate_label_model_multiclass(self):
    #     n1 = 5
    #     n2 = 3
    #     k = 3

    #     label_accuracies = np.array([[.9, .8, .5],
    #                                  [.6, .7, .3],
    #                                  [.6, .6, .8],
    #                                  [.7, .6, .6],
    #                                  [.8, .8, .9]])
    #     link_accuracies = np.array([.8, .6, .8])
    #     label_propensities = np.array([.9] * n1)
    #     link_propensities = np.array([.9] * n1)
    #     start_balance = np.array([.3, .3, .4])
    #     transitions = np.array([[.5, .3, .2],
    #                             [.4, .3, .3],
    #                             [.3, .3, .4]])

    #     labels, links, seq_starts, gold = _generate_data(
    #         1000, 8, 12, n1, n2,
    #         label_accuracies,
    #         link_accuracies,
    #         label_propensities,
    #         link_propensities,
    #         start_balance,
    #         transitions
    #     )

    #     model = LinkedHMM(k, n1, n2, acc_prior=0.0, balance_prior=0.0)
    #     config = LearningConfig()
    #     config.epochs = 4
    #     model.estimate_label_model(labels, links, seq_starts, config=config)

    #     for i in range(n1):
    #         for j in range(k):
    #             diff = label_accuracies[i, j] - model.get_accuracies()[i, j]
    #             self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(n2):
    #         for j in range(k):
    #             diff = link_accuracies[i] - model.get_link_accuracies()[i]
    #             self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(n1):
    #         diff = label_propensities[i] - model.get_propensities()[i]
    #         self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(n2):
    #         diff = link_propensities[i] - model.get_link_propensities()[i]
    #         self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(k):
    #         diff = start_balance[i] - model.get_start_balance()[i]
    #         self.assertAlmostEqual(diff, 0.0, places=1)
    #     for i in range(k):
    #         for j in range(k):
    #             diff = transitions[i, j] - model.get_transition_matrix()[i, j]
    #             self.assertAlmostEqual(diff, 0.0, places=1)

    # def test_get_most_probable_labels(self):
    #     m = 500
    #     n1 = 3
    #     n2 = 5
    #     k = 3

    #     model = LinkedHMM(k, n1, n2)
    #     with torch.no_grad():
    #         model.start_balance[0] = 0
    #         model.start_balance[1] = 0.5
    #         for i in range(n1):
    #             model.propensity[i] = 0
    #             for j in range(k):
    #                 model.accuracy[i, j] = 1
    #         for i in range(n2):
    #             model.link_propensity[i] = 0
    #             model.link_accuracy[i] = 1.5
    #         for i in range(k):
    #             for j in range(k):
    #                 model.transitions[i, j] = 1 if i == j else 0

    #     labels, links, seq_starts, gold = _generate_data(
    #         m, 8, 12, n1, n2,
    #         model.get_label_accuracies(),
    #         model.get_link_accuracies(),
    #         model.get_label_propensities(),
    #         model.get_link_propensities(),
    #         model.get_start_balance(),
    #         model.get_transition_matrix())

    #     predictions = model.get_most_probable_labels(labels, links, seq_starts)
    #     correct = 0
    #     for i in range(len(predictions)):
    #         if predictions[i] == gold[i]:
    #             correct += 1
    #     accuracy = correct / float(len(predictions))
    #     self.assertGreaterEqual(accuracy, .95)

    # def test_get_label_distribution(self):
    #     m = 500
    #     n1 = 3
    #     n2 = 5
    #     k = 3

    #     model = LinkedHMM(k, n1, n2)
    #     with torch.no_grad():
    #         model.start_balance[0] = 0
    #         model.start_balance[1] = 0.5
    #         for i in range(n1):
    #             model.propensity[i] = 0
    #             for j in range(k):
    #                 model.accuracy[i, j] = 1
    #         for i in range(n2):
    #             model.link_propensity[i] = 0
    #             model.link_accuracy[i] = 1.5
    #         for i in range(k):
    #             for j in range(k):
    #                 model.transitions[i, j] = 1 if i == j else 0

    #     labels, links, seq_starts, gold = _generate_data(
    #         m, 8, 12, n1, n2,
    #         model.get_label_accuracies(),
    #         model.get_link_accuracies(),
    #         model.get_label_propensities(),
    #         model.get_link_propensities(),
    #         model.get_start_balance(),
    #         model.get_transition_matrix())

    #     p_unary, p_pairwise = model.get_label_distribution(
    #         labels, links, seq_starts)

    #     # Makes predictions using both unary and pairwise marginals
    #     pred_unary = np.argmax(p_unary, axis=1) + 1
    #     pred_pairwise = np.zeros((labels.shape[0],), dtype=np.int)
    #     next_seq = 0
    #     for i in range(labels.shape[0] - 1):
    #         if next_seq == len(seq_starts) or i < seq_starts[next_seq] - 1:
    #             # i is neither the start nor end of a sequence
    #             pred_pairwise[i+1] = np.argmax(p_pairwise[i][pred_pairwise[i]])
    #         elif i == seq_starts[next_seq]:
    #             # i is the start of a sequence
    #             a, b = np.unravel_index(p_pairwise[i].argmax(), (k, k))
    #             pred_pairwise[i], pred_pairwise[i + 1] = a, b
    #             next_seq += 1
    #         else:
    #             # i is the end of a sequence
    #             pass
    #     pred_pairwise += 1

    #     # Checks that predictions are accurate
    #     for predictions in (pred_unary, pred_pairwise):
    #         correct = 0
    #         for i in range(len(predictions)):
    #             if predictions[i] == gold[i]:
    #                 correct += 1
    #         accuracy = correct / float(len(predictions))
    #         self.assertGreaterEqual(accuracy, .95)

    def test_get_k_most_probable_labels(self):
            m = 34 # num_seqs
            n1 = 4 # num_labeling_funcs
            n2 = 5 # num_linking_funcs
            k = 4 # num_classes

            model = LinkedHMM(k, n1, n2)
            with torch.no_grad():
                model.start_balance[0] = 0
                model.start_balance[1] = 0.5
                for i in range(n1):
                    model.propensity[i] = 0
                    for j in range(k):
                        model.accuracy[i, j] = 1
                for i in range(n2):
                    model.link_propensity[i] = 0
                    model.link_accuracy[i] = 1.5
                for i in range(k):
                    for j in range(k):
                        model.transitions[i, j] = 1 if i == j else 0

            labels, links, seq_starts, gold = _generate_data(
                m, 7, 7, n1, n2,
                model.get_label_accuracies(),
                model.get_link_accuracies(),
                model.get_label_propensities(),
                model.get_link_propensities(),
                model.get_start_balance(),
                model.get_transition_matrix())

            # model.estimate_label_model(labels, links, seq_starts)
            # # assert that when topk = 1, the output of get_k most_probable_labels is the same as get_most_probable_labels
            # # for the fact that torch.argmax (used in get_most_probable_labels) == torch.topk (used in get_k_most_probable_labels) 
            # # when topk = 1. 
            # predictions = model.get_most_probable_labels(labels, links, seq_starts)
            # k_predictions = model.get_k_most_probable_labels(labels, links, seq_starts, topk=1)
            # self.assertIsNone(np.testing.assert_array_equal(k_predictions[0], predictions)) 

            # # assert that when topk > 1, the viterbi_scores of the first sequence from get_k most_probable_labels 
            # # is the same as get_most_probable_labels
            # viterbi_scores = model.get_most_probable_labels(labels, links, seq_starts, return_viterbi_scores=True)
            # k_viterbi_scores = model.get_k_most_probable_labels(labels, links, seq_starts, topk=9, return_viterbi_scores=True)
            # self.assertIsNone(np.testing.assert_array_equal(k_viterbi_scores[:, 0], viterbi_scores))

            # # assert that when topk > 1, all sequences from get_k most_probable_labels are different from one another
            # k_predictions = model.get_k_most_probable_labels(labels, links, seq_starts, topk=3)
            # self.assertEqual(np.unique(k_predictions, axis=0).shape[0], k_predictions.shape[0])

            # # assert that when topk > 1, the viterbi scores are in a non-increasing order.

            ### 👀 Ontonotes
            labels, links, seq_starts = torch.load("/Users/zhengxinyong/Desktop/labelmodels/downloads/link_hmm_inputs.pt")
            print(labels.shape, links.shape) #### change num_labeling_funcs according given NoABS labeling function
            acc_prior = 50
            link_hmm = LinkedHMM(
                num_classes=4,
                num_labeling_funcs=8, 
                num_linking_funcs=7,
                init_acc=0.7,
                acc_prior=acc_prior,
                balance_prior=100)
            # link_hmm.estimate_label_model(labels, links, seq_starts)
            link_hmm_saved_fp = f"/Users/zhengxinyong/Desktop/labelmodels/ontonotes/labelmodel_link_hmm_prior_{acc_prior}.pt"
            # # torch.save(link_hmm, link_hmm_saved_fp)
            link_hmm = torch.load(link_hmm_saved_fp)
            print(f"✅ Done loading link hmm with acc_prior={acc_prior}.")
            print("get label accuracies:")
            print(link_hmm.get_label_accuracies())
            print("get label propensities:")
            print(link_hmm.get_label_accuracies())
            print("get link accuracies:")
            print(link_hmm.get_link_accuracies())
            print("get link propensities:")
            print(link_hmm.get_link_propensities())
            print("get start balance:")
            print(link_hmm.get_start_balance())
            print("get transition matrix:")
            print(link_hmm.get_transition_matrix())
            
            print(link_hmm.get_label_distribution(labels, links, seq_starts))

            # K = [5000]
            # for k in K:
            #     # viterbi_paths, viterbi_scores = link_hmm.get_k_most_probable_labels(labels[12066:12071, :], links[12066:12071, :], [0], topk=k, return_viterbi_scores=True)
            #     viterbi_paths, viterbi_scores = link_hmm.get_k_most_probable_labels(labels, links, seq_starts, topk=k, return_viterbi_scores=True)
            #     # print(seq_starts)
            #     # print(np.sum(np.exp(viterbi_scores), 0))
            #     print(f"✅ Done generating {k} (acc prior {acc_prior}).")
            #     print(viterbi_scores)
            #     torch.save(viterbi_paths, f"/Users/zhengxinyong/Desktop/labelmodels/ontonotes/{k}_viterbi_paths_prior_{acc_prior}.pt")
            #     torch.save(viterbi_scores, f"/Users/zhengxinyong/Desktop/labelmodels/ontonotes/{k}_viterbi_scores_prior_{acc_prior}.pt")
            
            # viterbi_paths = torch.load("/Users/zhengxinyong/Desktop/labelmodels/ontonotes/5_viterbi_paths.pt")
            # viterbi_scores = torch.load("/Users/zhengxinyong/Desktop/labelmodels/ontonotes/5_viterbi_scores.pt")
            # print(viterbi_scores)
            # print(viterbi_paths.shape)
            # print(viterbi_scores.shape)
            
            #### Ontonotes instance check: ensure that viterbi paths are correct (when topk > possible enumeration of sequences)
            # instance 780 only has 4 tokens
            # if k > 256:
            #     self.assertEqual(sum(viterbi_paths[256][seq_starts[780]:seq_starts[781]]), -4)
            #     self.assertEqual(viterbi_scores[256][780], -1)

            # instance 498, 561, 751, 769, 779, 784, 821, 822, 858 have 5 tokens

            # #### Ontonotes instance check: check scores
            # self.assertEqual(viterbi_scores[0][31], -112.71656799316406)
            # predictions, scores = link_hmm.get_most_probable_labels(labels, links, seq_starts, return_viterbi_scores=True)
            # self.assertEqual(scores[31], -112.71656799316406)

            # #### NECESSARY BUT INSUFFICIENT - comparison between get_most_probable and get_k_most_probable
            # self.assertIsNone(np.testing.assert_array_equal(viterbi_paths[0, :], predictions)) 
            # self.assertIsNone(np.testing.assert_array_equal(viterbi_scores[0, :], scores)) 


            # #### 💻 Laptop Reviews
            # labels, links, seq_starts = torch.load("/Users/zhengxinyong/Desktop/labelmodels/downloads/laptop_link_hmm_inputs.pt")
            # # print(labels.shape, links.shape) #### change num_labeling_funcs according given NoABS labeling function
            # link_hmm_saved_fp = f"/Users/zhengxinyong/Desktop/labelmodels/downloads/laptop_esteban_link_hmm.pt"
            # link_hmm = torch.load(link_hmm_saved_fp)
            # print("get label accuracies:")
            # print(link_hmm.get_label_accuracies())
            # print("get label propensities:")
            # print(link_hmm.get_label_accuracies())
            # print("get link accuracies:")
            # print(link_hmm.get_link_accuracies())
            # print("get link propensities:")
            # print(link_hmm.get_link_propensities())
            # print("get start balance:")
            # print(link_hmm.get_start_balance())
            # print("get transition matrix:")
            # print(link_hmm.get_transition_matrix())
            # print(f"smallest number of tokens in a sequence: {min([seq_starts[i + 1] - seq_starts[i] for i in range(len(seq_starts) - 1)])}")
            # print(f"largest number of tokens in a sequence: {max([seq_starts[i + 1] - seq_starts[i] for i in range(len(seq_starts) - 1)])}")

            # K = [1, 2, 3, 4, 5]
            # for k in K:
            #     viterbi_paths, viterbi_scores = link_hmm.get_k_most_probable_labels(labels, links, seq_starts, topk=k, return_viterbi_scores=True)
            #     torch.save(viterbi_paths, f"/Users/zhengxinyong/Desktop/labelmodels/laptop/{k}_viterbi_paths.pt")
            #     torch.save(viterbi_scores, f"/Users/zhengxinyong/Desktop/labelmodels/laptop/{k}_viterbi_scores.pt")
            #     print(f"✅ Laptop Reviews: Done generating {k}.")

            # predictions, scores = link_hmm.get_most_probable_labels(labels, links, seq_starts, return_viterbi_scores=True)
            # viterbi_paths = torch.load(f"/Users/zhengxinyong/Desktop/labelmodels/laptop/10_viterbi_paths.pt")

            # self.assertIsNone(np.testing.assert_array_equal(viterbi_paths[0, :], predictions)) 
            # print(-1 in viterbi_paths)

            # #### 🥼 NCBI
            # labels, links, seq_starts = torch.load("/Users/zhengxinyong/Desktop/labelmodels/downloads/ncbi_link_hmm_inputs.pt")
            # # print(labels.shape, links.shape) #### change num_labeling_funcs according given NoABS labeling function
            # link_hmm_saved_fp = f"/Users/zhengxinyong/Desktop/labelmodels/downloads/ncbi_esteban_link_hmm.pt"
            # link_hmm = torch.load(link_hmm_saved_fp)
            # print("get label accuracies:")
            # print(link_hmm.get_label_accuracies())
            # print("get label propensities:")
            # print(link_hmm.get_label_accuracies())
            # print("get link accuracies:")
            # print(link_hmm.get_link_accuracies())
            # print("get link propensities:")
            # print(link_hmm.get_link_propensities())
            # print("get start balance:")
            # print(link_hmm.get_start_balance())
            # print("get transition matrix:")
            # print(link_hmm.get_transition_matrix())

            # print(f"smallest number of tokens in a sequence: {min([seq_starts[i + 1] - seq_starts[i] for i in range(len(seq_starts) - 1)])}")
            # print(f"largest number of tokens in a sequence: {max([seq_starts[i + 1] - seq_starts[i] for i in range(len(seq_starts) - 1)])}")

            # for i in range(len(seq_starts) - 1):
            #     print(seq_starts[i + 1] - seq_starts[i], seq_starts[i], seq_starts[i + 1])

            # K = [1, 2, 3, 4, 5]
            # for k in K:
            #     viterbi_paths, viterbi_scores = link_hmm.get_k_most_probable_labels(labels, links, seq_starts, topk=k, return_viterbi_scores=True)
            #     torch.save(viterbi_paths, f"/Users/zhengxinyong/Desktop/labelmodels/ncbi/{k}_viterbi_paths.pt")
            #     torch.save(viterbi_scores, f"/Users/zhengxinyong/Desktop/labelmodels/ncbi/{k}_viterbi_scores.pt")
            #     print(f"✅ NCBI: Done generating {k}.")

            # viterbi_paths = torch.load(f"/Users/zhengxinyong/Desktop/labelmodels/ncbi/1_viterbi_paths.pt")
            # # print(viterbi_paths[0, 1242:1316])
            # # predictions, scores = link_hmm.get_most_probable_labels(labels, links, seq_starts, return_viterbi_scores=True)
            # self.assertIsNone(np.testing.assert_array_equal(viterbi_paths[0, :], predictions)) 
            
            
            # #### 💿 CDR
            # labels, links, seq_starts = torch.load("/Users/zhengxinyong/Desktop/labelmodels/downloads/cdr_link_hmm_inputs.pt")
            # # print(labels.shape, links.shape) #### change num_labeling_funcs according given NoABS labeling function
            # link_hmm_saved_fp = f"/Users/zhengxinyong/Desktop/labelmodels/downloads/cdr_esteban_link_hmm.pt"
            # link_hmm = torch.load(link_hmm_saved_fp)
            # print("get label accuracies:")
            # print(link_hmm.get_label_accuracies())
            # print("get label propensities:")
            # print(link_hmm.get_label_accuracies())
            # print("get link accuracies:")
            # print(link_hmm.get_link_accuracies())
            # print("get link propensities:")
            # print(link_hmm.get_link_propensities())
            # print("get start balance:")
            # print(link_hmm.get_start_balance())
            # print("get transition matrix:")
            # print(link_hmm.get_transition_matrix())

            # print(f"smallest number of tokens in a sequence: {min([seq_starts[i + 1] - seq_starts[i] for i in range(len(seq_starts) - 1)])}")
            # print(f"largest number of tokens in a sequence: {max([seq_starts[i + 1] - seq_starts[i] for i in range(len(seq_starts) - 1)])}")

            # K = [1, 2, 3, 4, 5]
            # for k in K:
            #     viterbi_paths, viterbi_scores = link_hmm.get_k_most_probable_labels(labels, links, seq_starts, topk=k, return_viterbi_scores=True)
            #     torch.save(viterbi_paths, f"/Users/zhengxinyong/Desktop/labelmodels/cdr/{k}_viterbi_paths.pt")
            #     torch.save(viterbi_scores, f"/Users/zhengxinyong/Desktop/labelmodels/cdr/{k}_viterbi_scores.pt")
            #     print(f"✅ BC5CDR: Done generating {k}.")

            # viterbi_paths = torch.load(f"/Users/zhengxinyong/Desktop/labelmodels/cdr/10_viterbi_paths.pt")
            # predictions, scores = link_hmm.get_most_probable_labels(labels, links, seq_starts, return_viterbi_scores=True)
            # self.assertIsNone(np.testing.assert_array_equal(viterbi_paths[0, :], predictions)) 

            # print(-1 in viterbi_paths)
            
            

    # def test_compute_viterbi(self):
    #     m = 1 # num_seqs
    #     n1 = 4 # num_labeling_funcs
    #     n2 = 5 # num_linking_funcs
    #     k = 3 # num_classes

    #     model = LinkedHMM(k, n1, n2)
    #     with torch.no_grad():
    #         model.start_balance[0] = 0
    #         model.start_balance[1] = 0.5
    #         for i in range(n1):
    #             model.propensity[i] = 0
    #             for j in range(k):
    #                 model.accuracy[i, j] = 1
    #         for i in range(n2):
    #             model.link_propensity[i] = 0
    #             model.link_accuracy[i] = 1.5
    #         for i in range(k):
    #             for j in range(k):
    #                 model.transitions[i, j] = 1 if i == j else 0

    #     labels, links, seq_starts, gold = _generate_data(
    #         m, 20, 20, n1, n2,
    #         model.get_label_accuracies(),
    #         model.get_link_accuracies(),
    #         model.get_label_propensities(),
    #         model.get_link_propensities(),
    #         model.get_start_balance(),
    #         model.get_transition_matrix())
        
    #     predictions = model.get_most_probable_labels(labels, links, seq_starts)
    #     scores = model.get_most_probable_labels(labels, links, seq_starts, return_viterbi_scores=True)
        
    #     path_scores_list = model.compute_viterbi(labels, links, seq_starts)

    #     for i in range(len(seq_starts)):
    #         start_idx = seq_starts[i]
    #         if i == len(seq_starts) - 1:
    #             end_idx = len(predictions)
    #         else:
    #             end_idx = seq_starts[i + 1]
    #         path = ''.join(map(str, list(predictions[start_idx:end_idx]-1)))
    #         print(path, scores[i], path_scores_list[i][path])
    #         assert scores[i] == path_scores_list[i][path]


    #     # ### TODO: test get_k_most_probable_labels
    #     # predictions = model.get_k_most_probable_labels(labels, links, seq_starts, topk=2)
    #     # scores = model.get_most_probable_labels(labels, links, seq_starts, return_viterbi_scores=True)
    #     # path_scores_list = model.compute_viterbi(labels, links, seq_starts)
        


def _generate_data(num_seqs, min_seq, max_seq, num_label_funcs, num_link_funcs,
                   label_accs, link_accs, label_propensities, link_propensities,
                   start_balance, transitions):
    # Generates sequence starts
    seq_starts = np.zeros((num_seqs,), dtype=int)
    total_len = 0
    for i in range(num_seqs):
        seq_len = np.random.randint(min_seq, max_seq + 1)
        total_len += seq_len
        if i + 1 < num_seqs:
            seq_starts[i + 1] = total_len

    # Generates sequences of gold labels
    gold = np.zeros((total_len,), dtype=int)
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
