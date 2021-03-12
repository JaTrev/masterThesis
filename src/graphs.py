import networkx as nx
from stellargraph import StellarGraph
import matplotlib.pyplot as plt
from src.misc import *

plt.rcParams['figure.figsize'] = [16, 9]


def remove_edges(graph: nx.Graph, edge_weights: list, percentile_cutoff: int, remove_isolated_nodes:bool):
    # remove edges that do not have a high enough similarity score
    min_cutoff_value = np.percentile(edge_weights, percentile_cutoff)
    # min(heapq.nlargest(percentile_cutoff, edge_weights))

    graph_edge_weights = nx.get_edge_attributes(graph, "weight")

    edges_to_kill = []
    for edge in graph.edges():
        edge_weight = graph_edge_weights[edge]

        if edge_weight < min_cutoff_value:
            edges_to_kill.append(edge)

    for edge in edges_to_kill:
        graph.remove_edge(edge[0], edge[1])

    if remove_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph


def create_networkx_graph(words: list, word_embeddings: list, similarity_threshold: float = 0.4,
                          percentile_cutoff: int = 70, remove_isolated_nodes: bool = True) -> nx.Graph:
    """
    create_networdx_graph creates a graph given the words and their embeddings
    :param words: list of words which will be the nodes
    :param word_embeddings: embeddings of the words
    :param similarity_threshold: cosine similarity threshold value for the edges
    :param percentile_cutoff: percentile threshold value
    :param remove_isolated_nodes: boolean indicating if isolated nodes should be removed
    :return: graph
    """
    assert len(words) == len(word_embeddings), "words and word_embeddings must have the same length"
    graph = nx.Graph()  # undirected graph
    # n = 0
    edge_weights = []

    first_half = word_embeddings[:int(len(word_embeddings)/2)]
    first_half_length = int(len(word_embeddings)/2)
    second_half = word_embeddings[int(len(word_embeddings)/2):]

    sim_matrix = cosine_similarity(first_half, second_half)
    for i in range(len(first_half)):
        word_i = words[i]

        for j in range(len(second_half)):
            word_j = words[first_half_length + j]

            sim = sim_matrix[i][j]

            if sim < similarity_threshold:
                # similarity is not high enough
                continue
            graph.add_edge(word_i, word_j, weight=float(sim))
            edge_weights.append(sim)

    return remove_edges(graph, edge_weights, percentile_cutoff, remove_isolated_nodes)


def create_networkx_graph_2(words: list, word_embeddings: list, similarity_threshold: float = 0.4,
                          percentile_cutoff: int = 80, remove_isolated_nodes: bool = True):
    """
    create_networdx_graph creates a graph given the words and their embeddings
    :param words: list of words which will be the nodes
    :param word_embeddings: embeddings of the words
    :param similarity_threshold: cosine similarity threshold value for the edges
    :param percentile_cutoff: percentile threshold value
    :param remove_isolated_nodes: boolean indicating if isolated nodes should be removed
    :return: graph
    """
    assert len(words) == len(word_embeddings), "words and word_embeddings must have the same length"
    graph = nx.Graph()  # undirected graph

    # nodes index -> index used in words to get the word
    graph.add_nodes_from(list(range(len(words))))
    n = 0
    edge_weights = []

    for i, word_i in enumerate(words):
        for j, word_j in enumerate(words):

            if i != j:
                if not (graph.has_edge(j, i)):

                    sim = cosine_similarity(word_embeddings[i].reshape(1, -1),
                                            word_embeddings[j].reshape(1, -1))

                    if sim < similarity_threshold:
                        # similarity is not high enough
                        continue

                    weight = sim
                    graph.add_edge(i, j, weight=float(weight))
                    edge_weights.append(weight)
                    n = n + 1

    new_graph = remove_edges(graph, edge_weights, percentile_cutoff, remove_isolated_nodes)
    return new_graph


def create_graph_with_features(graph: nx.Graph, node_list: list, feature_list: list, node_features: str = "feature"):

    assert len(node_list) == len(feature_list), "the feature_list must have a feature for each node"

    feature_graph = graph.copy()
    for node_id, node_data in feature_graph.nodes(data=True):
        node_index = node_list.index(node_id)
        node_data[node_features] = feature_list[node_index]

    return feature_graph


def networkx_to_stellargraph(feature_graph: nx.Graph, feature_name: str = "feature") -> StellarGraph:
    """
    networkx_to_stellargraph transforms a networkx graph to a stellargraph

    :param feature_graph: networkx graph
    :param feature_name: name of the node feature in this graph
    :return: stellargraph
    """
    return StellarGraph.from_networkx(feature_graph, node_features=feature_name)


def sort_words_by(graph: nx.Graph, word: str, word_weights: dict):

    neighbor_weights = []
    for w_neighbor in graph.adj[word]:
        neighbor_weights.append(float(graph.adj[word][w_neighbor]['weight']))

    sim_score = np.average(neighbor_weights)
    w_degree = graph.degree(word)
    w_weight = word_weights[word]

    return w_degree, sim_score, w_weight


def graph_evaluation_visualisation(graph: nx.Graph, processed_data: list, vocab: list, word_rank_type: str = "tf",
                                   k_component: int = 1, min_topic_number: int = 6) -> (list, plt):
    try:
        components = nx.k_components(graph)[k_component]
    except KeyError:
        return [["xx"]], None

    corpus_clusters = []
    for comp in components:
        if len(comp) >= min_topic_number:
            corpus_clusters.append(comp)

    cluster_words = []
    n_words = len([w for d in processed_data for w in d])
    word_weights = get_word_weights(processed_data, vocab, n_words, weight_type=word_rank_type)
    for i_c, c in enumerate(corpus_clusters):
        cluster_words.append(sorted(list(c), key=(lambda w: sort_words_by(graph, w, word_weights)), reverse=True))

    return cluster_words, plt
