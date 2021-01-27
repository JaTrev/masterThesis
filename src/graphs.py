import networkx as nx
import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from stellargraph import StellarGraph
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import GraphSAGELinkGenerator
import stellargraph.layer
from stellargraph.layer import GraphSAGE, link_classification, AttentionalAggregator,\
    MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator
from tensorflow import keras
from stellargraph.mapper import GraphSAGENodeGenerator
from math import sqrt
from src.misc import *
import heapq


plt.rcParams['figure.figsize'] = [16, 9]


def remove_edges(graph: nx.Graph, edge_weights: list, percentile_cutoff: int, remove_isolated_nodes):
    # remove edges that do not have a high enough similarity score
    min_cutoff_value = np.percentile(edge_weights, percentile_cutoff)
    # min(heapq.nlargest(percentile_cutoff, edge_weights))

    edges_to_kill = []
    for n, nbrs in graph.adj.items():

        for nbr, eattr in nbrs.items():

            # remove edges below a certain weight
            n_weight = eattr['weight']
            if n_weight < min_cutoff_value:
                edges_to_kill.append((n, nbr))

    for u, v in edges_to_kill:
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)

    if remove_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def create_networkx_graph(words: list, word_embeddings: list, similarity_threshold: float = 0.4,
                          percentile_cutoff: int = 80, remove_isolated_nodes: bool = True) -> nx.Graph:
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
    n = 0
    edge_weights = []

    for i, word_i in enumerate(words):
        for j, word_j in enumerate(words):

            if word_i != word_j:
                if not (graph.has_edge(word_j, word_i)):

                    sim = cosine_similarity(word_embeddings[i].reshape(1, -1),
                                            word_embeddings[j].reshape(1, -1))

                    if sim < similarity_threshold:
                        # similarity is not high enough
                        continue

                    weight = sim
                    graph.add_edge(word_i, word_j, weight=float(weight))
                    edge_weights.append(weight)
                    n = n + 1

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
        components_1 = nx.k_components(graph)[k_component]
    except KeyError:
        return [["xx"]], None

    corpus_clusters = []
    for comp in components_1:
        if len(comp) >= min_topic_number:
            corpus_clusters.append(comp)

    cluster_words = []
    n_words = len([w for d in processed_data for w in d])
    word_weights = get_word_weights(processed_data, vocab, n_words, weight_type=word_rank_type)
    for i_c, c in enumerate(corpus_clusters):
        cluster_words.append(sorted(list(c), key=(lambda w: sort_words_by(graph, w, word_weights)), reverse=True))

    return cluster_words, plt


def graph_sage_embeddings(sg_graph: StellarGraph, aggregator: stellargraph.layer = AttentionalAggregator,
                          batch_size: int = 50, number_of_walks: int = 5, length: int = 3, epochs: int = 10,
                          num_samples=[10, 5], optimizer: keras.optimizers = keras.optimizers.Adam(),
                          dropout: float = 0.0, layer_sizes=[50, 50], seed: int = 42) -> (list, list):

    nodes = list(sg_graph.nodes())

    generator = GraphSAGELinkGenerator(sg_graph, batch_size, list(num_samples), seed=seed)

    graph_sage = GraphSAGE(
        layer_sizes=list(layer_sizes), generator=generator, bias=True,
        dropout=dropout, normalize="l2", aggregator=aggregator)  # AttentionalAggregator: 0.72

    # Build the model and expose input and output sockets of graph_sage, for node pair inputs:
    x_inp, x_out = graph_sage.in_out_tensors()

    # classification layer
    prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="mul")(x_out)

    # stark GraphSAGE encoder and prediction layer
    model_sage = keras.Model(inputs=x_inp, outputs=prediction)

    model_sage.compile(
        optimizer=optimizer,
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy]
    )

    # train
    unsupervised_samples = UnsupervisedSampler(sg_graph, nodes=nodes, length=length,
                                               number_of_walks=number_of_walks, seed=seed)
    train_gen = generator.flow(unsupervised_samples, seed=seed)

    model_sage.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=1,
        shuffle=True,
    )

    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]

    # get new node embeddings
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    node_gen = GraphSAGENodeGenerator(sg_graph, batch_size, num_samples, seed=seed).flow(nodes, seed=seed)
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=1)

    corpus_node_embeddings = node_embeddings
    corpus_node_words = nodes
    assert len(corpus_node_embeddings) == len(corpus_node_words)

    return corpus_node_words, corpus_node_embeddings


def add_doc_to_graph(graph: nx.Graph, original_nodes: list, original_node_embeddings: list,
                     new_nodes: list, new_node_embeddings: list, new_node_features: list,
                     percentile_cutoff: int = 99, remove_isolated_nodes: bool = True):

    assert len(new_nodes) == len(new_node_embeddings), "need same amount of words and embeddings (new)"
    assert len(original_nodes) == len(original_node_embeddings), "need same amount of words and embeddings (original)"
    assert len(new_nodes) == len(new_node_features)

    new_graph = graph.copy()
    new_edges_weights = []

    # calculate similarity between new nodes and old ones
    for i, word_i in enumerate(new_nodes):
        for j, word_j in enumerate(original_nodes):
            if word_i != word_j:
                if not (new_graph.has_edge(word_j, word_i)):

                    sim = cosine_similarity(new_node_embeddings[i].reshape(1, -1),
                                            original_node_embeddings[j].reshape(1, -1))
                    if sim < 0:
                        continue
                    new_graph.add_edge(word_i, word_j, weight=sim)
                    new_edges_weights.append(sim)

    # calculate similarity between new nodes
    for i, word_i in enumerate(new_nodes):
        for j, word_j in enumerate(new_nodes):
            if word_i != word_j:
                if not (new_graph.has_edge(word_j, word_i)):

                    sim = cosine_similarity(new_node_embeddings[i].reshape(1, -1),
                                            new_node_embeddings[j].reshape(1, -1))
                    if sim < 0:
                        continue
                    new_graph.add_edge(word_i, word_j, weight=sim)
                    new_edges_weights.append(sim)

    new_graph = remove_edges(new_graph, new_edges_weights, percentile_cutoff, remove_isolated_nodes)

    new_feature_graph = create_graph_with_features(new_graph, new_nodes, new_node_features)

    return new_feature_graph


"""
    cluster_assignment = []
    for n in graph.nodes:

        found_cluster = False
        for i_c, c in enumerate(cluster_words):

            if n in c and not found_cluster:
                cluster_assignment.append(i_c + 1)
                found_cluster = True

        if not found_cluster:
            cluster_assignment.append(0)

    assert len(cluster_assignment) == len(graph.nodes)

    norm = mpl.colors.Normalize(vmin=0, vmax=len(cluster_words) + 1, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gist_ncar)

    count = graph.number_of_nodes()
    k = 10 / sqrt(count)
    pos = nx.fruchterman_reingold_layout(graph, k=k, iterations=300)

    nx.draw(graph, pos,
            nodelist=graph.nodes,
            font_size=15,
            node_size=40,
            edge_color='gray',
            node_color=[mapper.to_rgba(c_i) for c_i in cluster_assignment],
            with_labels=False)

    for p in pos:  # raise positions of the labels, relative to the nodes
        pos[p][1] -= 0.05
    nx.draw_networkx_labels(graph, pos, font_size=15, font_color='k')
    """