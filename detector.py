from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import numpy as np

def estimate_cumulative_probability_gmm(data, n_components=3):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data)

    cumulative_probabilities = {}

    for i, point in enumerate(data):
        prob = 0
        for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
            mvn = multivariate_normal(mean, covar, allow_singular=True)
            prob += weight * mvn.cdf(point)
        
        cumulative_probabilities[i] = prob

    return cumulative_probabilities

import networkx as nx

def cal_circle_feat_simple(graph_list):
    comm_cycles = []
    cycles_overlap_ratio = []

    for graph in graph_list:
        cycles = nx.cycle_basis(graph)
        comm_cycles.append(len(cycles))
        if len(cycles) == 0:
            cycles_overlap_ratio.append(0)
            continue 
        all_cycle_node = [node for cycle in cycles for node in cycle]
        overlap = (len(all_cycle_node) - len(set(all_cycle_node))) / len(set(all_cycle_node))
        cycles_overlap_ratio.append(overlap)

    return [comm_cycles, cycles_overlap_ratio]

def evaluate_node_level(topk, candi_cc, prob_df_sorted, true_node):
    pred_node = [candi_cc[c] for c in prob_df_sorted['component_id'].values[:topk]]
    pred_node = set([node for cc in pred_node for node in cc])

    recall = len(true_node & pred_node) / len(true_node)
    precision = len(true_node & pred_node) / len(pred_node)
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return f1, recall, precision

from sklearn.metrics import roc_auc_score
def cal_auc(graph_data, prob_df_gmm_sorted, candi_cc):
    true_label = np.zeros(len(graph_data.x))
    true_label[graph_data.y] = 1

    pred_label = np.zeros(len(graph_data.x))
    for i, c, s in prob_df_gmm_sorted.itertuples():
        for node in candi_cc[c]:
            if pred_label[node] == 0:
                pred_label[node] = s

    auc_score = roc_auc_score(true_label, pred_label)

    return auc_score