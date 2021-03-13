import networkx as nx
from src.misc import *

plt.rcParams['figure.figsize'] = [16, 9]


def remove_edges(graph: nx.Graph, edge_weights: list, percentile_cutoff: int, remove_isolated_nodes: bool):
    """

    :param graph:
    :param edge_weights:
    :param percentile_cutoff:
    :param remove_isolated_nodes:
    :return:
    """
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


def sort_words_by(graph: nx.Graph, word: str, word_weights: dict):
    """

    :param graph:
    :param word:
    :param word_weights:
    :return:
    """

    neighbor_weights = []
    for w_neighbor in graph.adj[word]:
        neighbor_weights.append(float(graph.adj[word][w_neighbor]['weight']))

    sim_score = np.average(neighbor_weights)
    w_degree = graph.degree(word)
    w_weight = word_weights[word]

    return w_degree, sim_score, w_weight
