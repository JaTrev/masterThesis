import networkx as nx
import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import stellargraph as sg
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification, AttentionalAggregator,\
    MeanPoolingAggregator, MaxPoolingAggregator, MeanAggregator
from tensorflow import keras
from stellargraph.mapper import GraphSAGENodeGenerator
from math import sqrt


plt.rcParams['figure.figsize'] = [16, 9]

def create_networkx_graph(words: list, word_embeddings: list, similarity_threshold: int = 0.4, percentile:int = 80,
                          remove_isolated_nodes:bool = True) -> nx.Graph:
    """
    create_networdx_graph creates a graph given the words and their embeddings
    :param words: list of words which will be the nodes
    :param word_embeddings: embeddings of the words
    :param similarity_threshold: cosine similarity threshold value for the edges
    :param percentile: percentile threshold value
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

    percentile_threshold = np.percentile(edge_weights, percentile)
    edges_to_kill = []

    for n, nbrs in graph.adj.items():

        for nbr, eattr in nbrs.items():

            # remove edges below a certain weight
            n_weight = eattr['weight']

            if n_weight < percentile_threshold:
                edges_to_kill.append((n, nbr))

    for u, v in edges_to_kill:
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)

    if remove_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    return graph


def add_node_feature(graph: nx.Graph, feature_list: list, node_feature:str = "feature"):

    assert len(graph.nodes()) == len(feature_list), "the feature_list must have a feature for each node"

    feature_graph = graph.copy()
    node_list = list(feature_graph.nodes())

    for node_id, node_data in feature_graph.nodes(data=True):
        node_index = node_list.index(node_id)
        node_data[node_feature] = feature_list[node_index]

    return feature_graph


def networkx_to_stellargraph(feature_graph: nx.Graph, feature_name: str = "feature") -> sg.StellarGraph:
    """
    networkx_to_stellargraph transforms a networkx graph to a stellargraph

    :param feature_graph: networkx graph
    :param feature_name: name of the node feature in this graph
    :return: stellargraph
    """
    return sg.from_networkx(feature_graph, node_features=feature_name)


def sort_words_by(graph: nx.Graph, word: str, word_counter: Counter):

    word_weights = []
    for w_neighbor in graph.adj[word]:
        word_weights.append(float(graph.adj[word][w_neighbor]['weight']))

    sim_score = np.average(word_weights)
    w_degree = graph.degree(word)
    w_tf = word_counter.get(word)

    return w_degree, sim_score, w_tf


def graph_evaluation(graph: nx.Graph, processed_data: list, k_component: int = 1,
                     min_topic_number: int = 5) -> (list, plt):
    temp_list = []
    for l in processed_data:
        temp_list.extend(l)
    word_counter = Counter(temp_list)

    components_1 = nx.k_components(graph)[k_component]

    corpus_clusters = []
    for comp in components_1:
        if len(comp) >= min_topic_number:
            corpus_clusters.append(comp)

    cluster_words = []
    for i_c, c in enumerate(corpus_clusters):
        cluster_words.append(sorted(list(c), key=(lambda w: sort_words_by(graph, w, word_counter)), reverse=True))

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

    plt.show()
    return cluster_words, plt


def graph_sage_embeddings(stellargraph: sg.StellarGraph, batch_size: int = 50,
                          number_of_walks: int = 5, length: int = 3, epochs: int = 10, num_samples=(10, 5)) \
        -> (list, list):

    nodes = list(stellargraph.nodes())

    generator = GraphSAGELinkGenerator(stellargraph, batch_size, [num_samples[0], num_samples[1]], seed=42)

    layer_sizes = [50, 50]

    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True,
        dropout=0.0, normalize="l2", aggregator=AttentionalAggregator)  # AttentionalAggregator: 0.72

    # Build the model and expose input and output sockets of graphsage, for node pair inputs:
    x_inp, x_out = graphsage.in_out_tensors()

    # classification layer
    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="mul")(x_out)

    # stark GraphSAGE encoder and prediction layer
    model_sage = keras.Model(inputs=x_inp, outputs=prediction)

    model_sage.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy]
    )

    # train
    unsupervised_samples = UnsupervisedSampler(stellargraph, nodes=nodes,
                                               length=length, number_of_walks=number_of_walks, seed=42)
    train_gen = generator.flow(unsupervised_samples, seed=42)

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
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    # node_ids = strong_G.nodes()
    node_gen = GraphSAGENodeGenerator(stellargraph, batch_size, num_samples, seed=42).flow(nodes, seed=42)
    node_embeddings = embedding_model.predict(node_gen, workers=1, verbose=1)

    corpus_node_embeddings = node_embeddings
    corpus_node_words = nodes
    assert len(corpus_node_embeddings) == len(corpus_node_words)

    return corpus_node_words, corpus_node_embeddings
