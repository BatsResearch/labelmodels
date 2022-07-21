import sys
mypath = "/Users/zhengxinyong/Desktop/labelmodels"
sys.path.append(mypath)


from labelmodels import LinkedHMM, LearningConfig
import numpy as np
from scipy import sparse
from scipy import special
import torch
from tqdm import tqdm

def sample_k_labels(input_fp, output_fp, dataset, k=1000):
    # TODO: integrate into LinkedHMM
    labels, links, seq_starts = torch.load(f"{input_fp}/{dataset}_link_hmm_inputs.pt")
    model_saved_fp = f"{input_fp}/{dataset}_link_hmm.pt"
    model = torch.load(model_saved_fp)
    print(f"✅ Done loading link hmm.")
    print("get label accuracies:")
    print(model.get_label_accuracies())

    # p_unary, p_pairwise = model.get_label_distribution(labels, links, seq_starts)
    # torch.save([p_unary, p_pairwise], f"{output_fp}/emp_dist/{dataset}_unary_pairwise.pt")
    p_unary, p_pairwise = torch.load(f"{output_fp}/emp_dist/{dataset}_unary_pairwise.pt")

    print(f"⛏ Sampling k={k} label sequences")
    paths = np.zeros((k, p_unary.shape[0]), dtype=int)
    instance_idx = -1
    num_choices = p_unary.shape[1]
    for i in tqdm(range(k)):
        for j in range(p_unary.shape[0]):
            if j in seq_starts:
                instance_idx += 1
                label = np.random.choice(num_choices, size=1, p=p_unary[j]/p_unary[j].sum())
                paths[i][j] = label[0] + 1
            else:
                prev_label = paths[i][j - 1] - 1
                next_label_dist = p_pairwise[j - 1][prev_label]
                label = np.random.choice(num_choices, size=1, p=next_label_dist/next_label_dist.sum())
                paths[i][j] = label[0] + 1
        instance_idx = -1

    torch.save(paths, f"{output_fp}/{dataset}_{k}_sampled_paths.pt")
    return paths

def sampling_empirical_distribution(input_fp, output_fp, dataset, k=100):
    paths = torch.load(f"{output_fp}/{dataset}_{k}_sampled_paths.pt")
    model_saved_fp = f"{input_fp}/{dataset}_link_hmm.pt"
    model = torch.load(model_saved_fp)
    empirical_dist = np.zeros((paths.shape[1], model.get_label_accuracies().shape[1]))
    print("empirical_dist.shape:", empirical_dist.shape)
    
    for path_k in tqdm(range(paths.shape[0])):
        path = paths[path_k]

        for i in range(len(path)):
            if path[i] < 0:
                continue
            
            empirical_dist[i][path[i] - 1] += 1
    empirical_dist = empirical_dist / k
    np.save(open(f"{output_fp}/emp_dist/{dataset}_sampled_{k}.npy", "wb"), empirical_dist)

def top_k_empirical_distribution(input_fp, output_fp, dataset, acc_prior=50, k=50):
    labels, links, seq_starts = torch.load(f"{input_fp}/{dataset}_link_hmm_inputs.pt")
    link_hmm_saved_fp = f"{input_fp}/labelmodel_link_hmm_prior_{acc_prior}.pt"
    link_hmm = torch.load(link_hmm_saved_fp)
    print(f"✅ Done loading link hmm with acc_prior={acc_prior}.")
    print("get label accuracies:")
    print(link_hmm.get_label_accuracies())
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

    # getting empirical distribution
    print(f"Getting empirical distribution (k = {k}):")
    viterbi_paths = torch.load(f"{output_fp}/{k}_viterbi_paths_prior_50.pt")
    viterbi_scores = torch.load(f"{output_fp}/{k}_viterbi_scores_prior_50.pt")
    viterbi_scores = np.ma.masked_where(viterbi_scores == -1, viterbi_scores)

    empirical_dist = np.zeros((viterbi_paths.shape[1], link_hmm.get_label_accuracies().shape[1]))
    print("empirical_dist.shape:", empirical_dist.shape)
    
    for path_k in tqdm(range(viterbi_paths.shape[0])):
        path = viterbi_paths[path_k]

        for i in range(len(path)):
            if path[i] < 0:
                continue

            if i in seq_starts:
                score = viterbi_scores[path_k][np.where(seq_starts == i)[0][0]]
                all_scores = viterbi_scores[:, np.where(seq_starts == i)[0][0]]
                all_scores = all_scores[all_scores.mask == False]
                total_score = special.logsumexp(all_scores.filled())
            
            empirical_dist[i][path[i] - 1] += np.exp(score - total_score)  # weighted by scores
    
    print(empirical_dist[17065:17069])
    np.save(open(f"{output_fp}/emp_dist/{dataset}_top_{k}.npy", "wb"), empirical_dist)

def get_posterior_marginal(dataset):
    # right now I treat unary marginal as posterior marginal
    ...


if __name__ == '__main__':
    # top_k_empirical_distribution(
    #     input_fp="/Users/zhengxinyong/Desktop/labelmodels/inputs/ontonotes", 
    #     output_fp="/Users/zhengxinyong/Desktop/labelmodels/outputs/ontonotes",
    #     dataset="ontonotes")
    

    for k in [2000, 3000, 10000]:
        sample_k_labels(
            input_fp="/Users/zhengxinyong/Desktop/labelmodels/inputs/ontonotes", 
            output_fp="/Users/zhengxinyong/Desktop/labelmodels/outputs/ontonotes",
            dataset="ontonotes",
            k=k)

        sampling_empirical_distribution(
            input_fp="/Users/zhengxinyong/Desktop/labelmodels/inputs/ontonotes", 
            output_fp="/Users/zhengxinyong/Desktop/labelmodels/outputs/ontonotes",
            dataset="ontonotes",
            k=k)
