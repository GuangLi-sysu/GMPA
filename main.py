import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import to_networkx

from communities import graphsage_cl
from split_util import (
    assign_persona_by_cluster,
    decompose_graph_by_persona_remove,
    get_connected_components,
    relabel_nodes
)
from detector import (
    cal_circle_feat_simple,
    evaluate_node_level,
    estimate_cumulative_probability_gmm,
    cal_auc
)

def load_graph_data(dataset_name):
    data_path = f'./data/{dataset_name}.pt'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    return torch.load(data_path)


def preprocess_graph_features(graph_data, origin_graph):
    degree_features = [deg for _, deg in origin_graph.degree()]
    degree_tensor = torch.tensor(degree_features, dtype=torch.float)
    normalized_features = (degree_tensor - degree_tensor.mean()) / degree_tensor.std()
    return normalized_features


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def filter_candidate_components(connected_components, original_graph_size):
    max_size = original_graph_size / 2
    candidates = {}
    
    for idx, component in enumerate(connected_components):
        component_size = len(component)
        if 1 < component_size < max_size:
            candidates[idx] = component
    
    return candidates


def process_dataset(dataset_name, top_k=5):
    print(f"\n{'='*50}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*50}\n")
    
    graph_data = load_graph_data(dataset_name)
    graph_data.edge_weight = graph_data.edge_weight.squeeze(0)
    graph_data.edge_timestamp = graph_data.edge_timestamp.squeeze(0)
    
    origin_graph = to_networkx(graph_data)
    origin_graph_undirected = origin_graph.to_undirected()
    
    degree_features = preprocess_graph_features(graph_data, origin_graph)
    
    device = get_device()
    graph_data = graph_data.to(device)
    degree_features = degree_features.reshape(-1, 1).to(device)
    
    embedding, model = graphsage_cl(
        x=degree_features,
        edge_index=graph_data.edge_index,
        embed_channel=32,
        iter_num=100,
        model_file=None,
        device=device
    )
    embedding = embedding.cpu()
    
    persona_mapping = assign_persona_by_cluster(embedding, origin_graph)
    decomposed_graph, node_id_mapping = decompose_graph_by_persona_remove(
        origin_graph, 
        persona_mapping
    )
    
    connected_components = get_connected_components(decomposed_graph)
    print(f"Connected components after decomposition: {len(connected_components)}")
    
    connected_components_relabeled = relabel_nodes(connected_components, node_id_mapping)
    
    candidate_components = filter_candidate_components(
        connected_components_relabeled,
        original_graph_size=graph_data.x.shape[0]
    )
    print(f"Candidate components: {len(candidate_components)}")
    
    subgraphs = [
        origin_graph_undirected.subgraph(component)
        for component in candidate_components.values()
    ]
    
    circle_features = cal_circle_feat_simple(subgraphs)
    feature_matrix = np.array(circle_features).T
    
    probabilities_gmm = estimate_cumulative_probability_gmm(feature_matrix)
    relabeled_probabilities = {
        comp_id: prob
        for comp_id, prob in zip(candidate_components.keys(), probabilities_gmm.values())
    }
    
    prob_df = pd.DataFrame({
        'component_id': list(relabeled_probabilities.keys()),
        'probability': list(relabeled_probabilities.values())
    })
    prob_df_sorted = prob_df.sort_values(by='probability', ascending=False)
    
    anomaly_nodes = set(graph_data.y)
    
    f1_score, recall, precision = evaluate_node_level(
        topk=top_k,
        candi_cc=candidate_components,
        prob_df_sorted=prob_df_sorted,
        true_node=anomaly_nodes
    )
    
    auc_score = cal_auc(graph_data, prob_df_sorted, candidate_components)
    
    print(f"\nResults for {dataset_name}:")
    print(f"  Top-{top_k} F1 Score: {f1_score:.4f}")
    print(f"  Top-{top_k} Recall: {recall:.4f}")
    print(f"  Top-{top_k} Precision: {precision:.4f}")
    print(f"  AUC Score: {auc_score:.4f}")
    
    return f1_score, recall, precision, auc_score


def main():
    datasets = ['Harmony_origin', 'Upbithack_origin']
    results = {}
    
    for dataset in datasets:
        try:
            metrics = process_dataset(dataset, top_k=5)
            results[dataset] = metrics
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
    
    if results:
        print(f"\n{'='*50}")
        print("SUMMARY OF RESULTS")
        print(f"{'='*50}")
        for dataset, (f1, recall, precision, auc) in results.items():
            print(f"\n{dataset}:")
            print(f"  F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUC: {auc:.4f}")


if __name__ == "__main__":
    main()