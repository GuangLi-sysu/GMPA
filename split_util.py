import networkx as nx

def get_connected_components(graph_data_nx):
    if nx.is_directed(graph_data_nx):
        return list(nx.weakly_connected_components(graph_data_nx))
    else:
        return list(nx.connected_components(graph_data_nx))


def relabel_nodes(all_cc, mapping):
    new_cc = []
    miss_map = 0
    for cc in all_cc:
        c = []
        for node in cc:
            if node in mapping.keys():
                c.append(mapping[node])
            else:
                miss_map += 1
                c.append(node)
        new_cc.append(list(set(c)))
    return new_cc

from sklearn.cluster import AgglomerativeClustering
def clustering(embeddings, node_id):
    clusterer = AgglomerativeClustering(distance_threshold=5, n_clusters=None)
    clusterer.fit(embeddings.detach().numpy())
    predicted_labels = clusterer.labels_

    cluster = [[] for i in range(max(predicted_labels)+1)]

    for node, c in zip(node_id, predicted_labels):
        cluster[c].append(node)

    return predicted_labels, cluster

def assign_persona_by_cluster(embedding, origin_graph):
    persona_mapping = []

    for node in origin_graph.nodes():
        node_neigh = list(origin_graph.neighbors(node))
        if len(node_neigh) < 2:
            continue
        node_neigh_embed = embedding[node_neigh]
        _, neigh_cluster = clustering(node_neigh_embed, node_neigh)
        if len(neigh_cluster) < 2:
            continue
        
        for cluster in neigh_cluster:
            persona_mapping.append((node, cluster))
    return persona_mapping

def decompose_graph_by_persona_remove(graph, persona_mapping):
    new_graph = graph.copy()
    index = graph.number_of_nodes()
    node_mapping = {}
    del_nodes = []
    del_edges = []

    for node, cluster in persona_mapping:
        in_neigh = set([source for source, target in graph.in_edges(node)])
        out_neigh = set([target for source, target in graph.out_edges(node)])
        for neigh in cluster:
            if neigh in in_neigh:
                new_graph.add_edge(neigh, index)
                del_edges.append((neigh, node))
            if neigh in out_neigh:
                new_graph.add_edge(index, neigh)
                del_edges.append((node, neigh))
        del_nodes.append(node)
        node_mapping[index] = node
        index += 1
    
    new_graph.remove_edges_from(del_edges)
    new_graph.remove_nodes_from(del_nodes)
    return new_graph, node_mapping

def decompose_graph_by_persona_isolate(graph, persona_mapping):
    new_graph = graph.copy()
    index = graph.number_of_nodes()
    node_mapping = {}
    del_nodes = []
    del_edges = []

    for node in graph.nodes():
        node_mapping[node] = node

    for node, cluster in persona_mapping:
        in_neigh = set([source for source, target in graph.in_edges(node)])
        out_neigh = set([target for source, target in graph.out_edges(node)])
        for neigh in cluster:
            if neigh in in_neigh:
                new_graph.add_edge(neigh, index)
                del_edges.append((neigh, node))
            if neigh in out_neigh:
                new_graph.add_edge(index, neigh)
                del_edges.append((node, neigh))
        del_nodes.append(node)
        node_mapping[index] = node
        index += 1
    print(f'Before decompose graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')
    
    new_graph.remove_edges_from(del_edges)
    isolates = list(nx.isolates(new_graph))

    new_graph.remove_nodes_from(isolates)
    print(f'After decompose graph: {new_graph.number_of_nodes()} nodes, {new_graph.number_of_edges()} edges')
    return new_graph, node_mapping
