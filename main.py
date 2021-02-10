from src.get_data import *
from src.preprocessing import *
from src.model import *
from src.evaluation import *
from src.clustering import *
from src.vectorization import *
from src.visualizations import *
from src.bert import *
from src.graphs import *
from src.doc_space import *
from src.w_d_space import *
from src.word_space import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from collections import Counter
import networkx as nx
import karateclub


data, data_labels, test_data, test_data_labels = get_data()

new_data = []
new_data_label = []
for i, d in enumerate(data):
    if len([w for w in d.split() if w.isalpha()]) > 2:
        new_data.append(d)
        new_data_label.append(data_labels[i])
print("removed docs: " + str(len(data) - len(new_data)))

new_test_data = []
new_test_data_label = []
for i, d in enumerate(test_data):

    if len([w for w in d.split() if w.isalpha()]) > 2:
        new_test_data.append(d)
        new_test_data_label.append(test_data_labels[i])
print("removed test docs: " + str(len(test_data) - len(new_test_data)))


# all_data = [d for d in new_data]
# all_data.extend(new_test_data)

# all_data_label = [l for l in new_data_label]
# all_data_label.extend(new_test_data_label)
# assert len(all_data) == len(all_data_label)


# TODO: create a main() function


def number_of_words_per_doc():
    fig, ax = plt.subplots(figsize=(10, 6))

    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.linewidth'] = 2

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')

    ax.yaxis.grid(color='grey', linestyle="--")
    ax.xaxis.grid(alpha=0)

    plt.margins(0)

    all_data_lengths = [len([w for w in doc.split() if w.isalpha()]) for doc in new_data]
    data_lengths_c = [all_data_lengths.count(int(i)) for i in range(int(np.max(all_data_lengths)))]
    plt.bar(range(int(np.max(all_data_lengths))), data_lengths_c, color="black")

    plt.xlim(right=int(np.max(all_data_lengths)))
    plt.xlim(left=0)

    plt.ylim(top=int(np.max(data_lengths_c)))
    plt.ylim(bottom=0)

    ax.set_xlabel("Number of Words", fontsize="medium")
    ax.set_ylabel("Number of Segments", fontsize="medium")

    plt.show()
    fig.savefig("visuals/segment_word_distribution.pdf", bbox_inches='tight', transparent=True)


def vis_most_common_words(data: list, raw_data: False, preprocessed: False):
    if raw_data:
        data = [doc.split() for doc in data]
        y_max = 25000
        filename = "most_common_words"
    else:
        if preprocessed:

            y_max = 1000
        else:

            y_max = 4000
        filename = "processed_most_common_words"

    fig, ax = plt.subplots(figsize=(10, 6))

    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 2

    ax.tick_params(axis='both', labelsize=12)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')

    ax.yaxis.grid(color='grey', linestyle="--", alpha=0.5)
    ax.xaxis.grid(alpha=0)
    plt.margins(0)

    data_words = []
    for doc in data:
        data_words.extend([w.lower() for w in doc if w.isalpha()])

    data_words_c = Counter(data_words)

    most_common_words = [w for w, c in data_words_c.most_common(30)]
    most_common_words_c = [c for w, c in data_words_c.most_common(30)]

    plt.bar(most_common_words, most_common_words_c, color='black', width=0.5)

    plt.ylim(top=y_max)
    plt.ylim(bottom=0)

    ax.set_xlabel("Top 30 Words", fontsize="medium")
    ax.set_ylabel("Number of Occurrences", fontsize="medium")

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    fig.savefig("visuals/" + str(filename) + ".pdf", bbox_inches='tight', transparent=True)



def doc2vec_visualization(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):
    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    d2v_params = {"size": 300,
                    "min_c": 15,
                    "win": 15,
                    "sample": 1e-5,
                    "negative": 0,
                    "hs": 1,
                    "epochs": 400,
                    "seed": 42}
    words, word_embeddings, _, _ = get_doc2vec_embeddings(tokenized_docs, vocab, **d2v_params)

    # get word embedding similarities
    # word_embeddings = get_word_similarities(word_embeddings)

    k_10_c_v = {"kmeans": 0, "agglomerative": 0, "spectral": 0, "nmf": 0, "hdbscan": 0}
    best_c_v = {"kmeans": 0, "agglomerative": 0, "spectral": 0, "nmf": 0, "hdbscan": 0}
    k_10_topics = {"kmeans": None, "agglomerative": None, "spectral": None, "nmf": None, "hdbscan": None}
    best_c_v_topics = {"kmeans": None, "agglomerative": None, "spectral": None, "nmf": None, "hdbscan": None}

    worst_c_v = {"kmeans": 1, "agglomerative": 1, "spectral": 1, "nmf": 1, "hdbscan": 1}
    worst_c_v_topics = {"kmeans": None, "agglomerative": None, "spectral": None, "nmf": None, "hdbscan": None}

    y_c_v_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": [], "nmf": [], "hdbscan": []}
    y_dbs_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": [], "nmf": [], "hdbscan": []}
    y_npmi_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": [], "nmf": [], "hdbscan": []}

    hdbscan_done = False
    hdbscan_words = None
    hdbscan_embedding = None

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "spectral", "nmf", "hdbscan"]:

            if cluster_type == "hdbscan":

                if not hdbscan_done:
                    labels, probabilities = hdbscan_clustering(word_embeddings, do_dim_reduction=True)

                    clusters_words_temp = [[] for _ in range(len(set(labels)) - 1)]
                    clusters_words_embeddings_temp = [[] for _ in range(len(set(labels)) - 1)]
                    for i_l, label in enumerate(labels):

                        if label == -1:
                            # noise
                            continue
                        clusters_words_temp[label].append([words[i_l], probabilities[i_l]])
                        clusters_words_embeddings_temp[label].append(word_embeddings[i_l])

                    clusters_words = []
                    clusters_words_embeddings = []
                    for i_c in range(len(clusters_words_temp)):
                        # c_new = sorted(c, key=lambda item: item[1], reverse=True)
                        c_words = clusters_words_temp[i_c]
                        c_embeddings = clusters_words_embeddings_temp[i_c]

                        c_order = sorted(range(len(c_words)), key=lambda index: c_words[index][1], reverse=True)

                        # new_cluster_words.append([w for w, _ in c_new[:10]])
                        clusters_words.append([c_words[i_order][0] for i_order in c_order[:10]])
                        clusters_words_embeddings.append([c_embeddings[i_order] for i_order in c_order[:10]])

                        hdbscan_words = clusters_words
                        hdbscan_embedding = clusters_words_embeddings

                    hdbscan_done = True

                else:
                    clusters_words = hdbscan_words
                    clusters_words_embeddings = hdbscan_embedding

            else:

                if cluster_type in ["kmeans", "nmf"]:
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                clusters_words, clusters_words_embeddings = word_clusters(
                    all_data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                    params=clustering_params, clustering_weight_type=clustering_weight_type,
                    ranking_weight_type=ranking_weight_type
                )
            print("cluster len")
            print(len(clusters_words))
            print("-----------------")
            print()
            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, clusters_words, cs_type='c_v')))
            dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
            cs_npmi = average_npmi_topics(all_data_processed, clusters_words, len(clusters_words))

            y_c_v_clustering_type[cluster_type].append(cs_c_v)
            y_npmi_clustering_type[cluster_type].append(cs_npmi)
            y_dbs_clustering_type[cluster_type].append(dbs)

            if cs_c_v > best_c_v[cluster_type]:
                best_c_v[cluster_type] = cs_c_v
                best_c_v_topics[cluster_type] = clusters_words

            if cs_c_v < worst_c_v[cluster_type]:
                worst_c_v[cluster_type] = cs_c_v
                worst_c_v_topics[cluster_type] = clusters_words

            if k == 10:
                k_10_c_v[cluster_type] = cs_c_v
                k_10_topics[cluster_type] = clusters_words

    print("best c_v scores:")
    for m, b_cs in best_c_v.items():
        print(str(m) + ": " + str(b_cs))

    print()
    print("k=10 c_v scores:")
    for m, cs_score in k_10_c_v.items():
        print(str(m) + ": " + str(cs_score))

    # c_v coherence score
    ys = [l for l in y_c_v_clustering_type.values()]
    print("---------------")
    print(ys)
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "Spectral", "NMF", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/c_v_d2v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_npmi_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="NPMI",
                          color_legends=["K-Means", "Agglomerative", "Spectral", "NMF", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/c_npmi_d2v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative", "Spectral", "NMF", "HDBSCAN"], type='dbs')
    fig.savefig("visuals/dbi_d2v_vs_k.pdf", bbox_inches='tight', transparent=True)

    best_c_v_topics_lengths = {"kmeans": None, "agglomerative": None, "spectral": None, "nmf": None, "hdbscan": None}
    for m, topics in best_c_v_topics.items():
        g, plt = create_circle_tree(topics)
        fig = plt.gcf()
        fig.savefig("visuals/best_" + str(m) + ".pdf", dpi=100, transparent=True)
        nx.write_graphml(g, "visuals/best_" + str(m) + ".graphml")

        # k = 10 model
        g, plt = create_circle_tree(k_10_topics[m])
        nx.write_graphml(g, "visuals/k=10_" + str(m) + ".graphml")

        # add to best_c_v_topics_lengths
        best_c_v_topics_lengths[m] = [len(t) for t in topics]

        # write topics
        write_topics_viz(topics, best_c_v[m], m,
                         "visuals/best_" + str(m) + ".txt")
        write_topics_viz(worst_c_v_topics[m], worst_c_v[m], m,
                         "visuals/worst_" + str(m) + ".txt")
        # write k = 10 model
        write_topics_viz(k_10_topics[m], k_10_c_v[m], m,
                         "visuals/k=10_" + str(m) + ".txt")

    best_topics_lengths = [l for l in best_c_v_topics_lengths.values()]
    _, fig = box_plot(best_topics_lengths, ["K-Means", "Agglomerative", "Spectral", "NMF", "HDBSCAN"],
                      "Clustering Types",  "Topic Lengths")
    fig.savefig("visuals/box_plot_d2v.pdf", dpi=100, transparent=True)





def fast_text_visualization(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):
    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    words, word_embeddings = get_fast_text_embeddings(all_data_processed, vocab)

    k_10_c_v = {"kmeans": 0, "agglomerative": 0}
    best_c_v = {"kmeans": 0, "agglomerative": 0}
    k_10_topics = {"kmeans": None, "agglomerative": None}
    best_c_v_topics = {"kmeans": None, "agglomerative": None}

    worst_c_v = {"kmeans": 1, "agglomerative": 1}
    worst_c_v_topics = {"kmeans": None, "agglomerative": None}

    y_c_v_clustering_type = {"kmeans": [], "agglomerative": []}
    y_dbs_clustering_type = {"kmeans": [], "agglomerative": []}
    y_u_mass_clustering_type = {"kmeans": [], "agglomerative": []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative"]:

            if cluster_type == "kmeans":
                clustering_params = {'n_clusters': k, 'random_state': 42, }
            else:
                clustering_params = {'n_clusters': k}

            clusters_words, clusters_words_embeddings = word_clusters(
                all_data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                params=clustering_params, clustering_weight_type=clustering_weight_type,
                ranking_weight_type=ranking_weight_type
            )

            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, clusters_words, cs_type='c_v')))
            dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
            cs_npmi = average_npmi_topics(all_data_processed, clusters_words, len(clusters_words))

            y_c_v_clustering_type[cluster_type].append(cs_c_v)
            y_u_mass_clustering_type[cluster_type].append(cs_npmi)
            y_dbs_clustering_type[cluster_type].append(dbs)

            if cs_c_v > best_c_v[cluster_type]:
                best_c_v[cluster_type] = cs_c_v
                best_c_v_topics[cluster_type] = clusters_words

            if cs_c_v < worst_c_v[cluster_type]:
                worst_c_v[cluster_type] = cs_c_v
                worst_c_v_topics[cluster_type] = clusters_words

            if k == 10:
                k_10_c_v[cluster_type] = cs_c_v
                k_10_topics[cluster_type] = clusters_words

    print("best c_v scores:")
    for m, b_cs in best_c_v.items():
        print(str(m) + ": " + str(b_cs))

    print()
    print("k=10 c_v scores:")
    for m, cs_score in k_10_c_v.items():
        print(str(m) + ": " + str(cs_score))

    # c_v coherence score
    ys = [l for l in y_c_v_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative"], type='c_v')
    fig.savefig("visuals/c_v_fastText_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_u_mass_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="NPMI",
                          color_legends=["K-Means", "Agglomerative"], type='c_npmi')
    fig.savefig("visuals/c_npmi_fastText_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative"], type='dbs')
    fig.savefig("visuals/dbi_fastText_vs_k.pdf", bbox_inches='tight', transparent=True)

    best_c_v_topics_lengths = {"kmeans": None, "agglomerative": None}
    for m, topics in best_c_v_topics.items():
        g, plt = create_circle_tree(topics)
        fig = plt.gcf()
        fig.savefig("visuals/best_" + str(m) + ".pdf", dpi=100, transparent=True)
        nx.write_graphml(g, "visuals/best_" + str(m) + ".graphml")

        # k = 10 model
        g, plt = create_circle_tree(k_10_topics[m])
        nx.write_graphml(g, "visuals/k=10_" + str(m) + ".graphml")

        # add to best_c_v_topics_lengths
        best_c_v_topics_lengths[m] = [len(t) for t in topics]

        # write topics
        write_topics_viz(topics, best_c_v[m], m,
                         "visuals/best_" + str(m) + ".txt")
        write_topics_viz(worst_c_v_topics[m], worst_c_v[m], m,
                         "visuals/worst_" + str(m) + ".txt")
        # write k = 10 model
        write_topics_viz(k_10_topics[m], k_10_c_v[m], m,
                         "visuals/k=10_" + str(m) + ".txt")

    best_topics_lengths = [l for l in best_c_v_topics_lengths.values()]
    _, fig = box_plot(best_topics_lengths, ["K-Means", "Agglomerative"], "Clustering Types", "Topic Lengths")
    fig.savefig("visuals/box_plot_fastText.pdf", dpi=100, transparent=True)


def karate_club(original_data, all_data_processed, vocab, tokenized_docs):

    vocab_words, vocab_embeddings, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")

    best_c_v = {1: 0, 2: 0, 3: 0}
    best_c_v_topics = {1: None, 2: None, 3: None}

    worst_c_v = {1: 1, 2: 1, 3: 1}
    worst_c_v_topics = {1: None, 2: None, 3: None}

    y_c_v_clustering_type = {1: [], 2: [], 3: []}
    y_dbs_clustering_type = {1: [], 2: [], 3: []}
    y_u_mass_clustering_type = {1: [], 2: [], 3: []}

    x = [0.4]

    for sim in x:

        graph = create_networkx_graph_2(vocab_words, vocab_embeddings, similarity_threshold=0, percentile_cutoff=50)

        node_sentence_embeddings, node_doc_embeddings = get_avg_sentence_doc_embeddings_w2v_2(original_data,
                                                                                              list(graph.nodes()),
                                                                                              vocab_words,
                                                                                              vocab_embeddings)
        node_features = []
        for node in graph.nodes():
            embedding = w2v_model.wv.vectors[w2v_model.wv.index2word.index(vocab_words[node])]
            sentence_embedding = node_sentence_embeddings[node]
            doc_embedding = node_doc_embeddings[node]

            node_feature = embedding.tolist()
            # node_feature.extend(sentence_embedding)

            node_features.append(node_feature)

        # feature_graph = create_graph_with_features(graph, list(graph.nodes()), node_features)

        model = karateclub.TENE()
        model.fit(graph, np.array(node_features))
        node_embeddings = model.get_embedding()
        node_words = [vocab_words[n] for n in graph.nodes]

        clusters_words, clusters_words_embeddings = word_clusters(
            all_data_processed, node_words, node_embeddings, vocab, clustering_type="kmeans",
            params={'n_clusters': 10, 'random_state': 42, }, clustering_weight_type='tf',
            ranking_weight_type='tf'
        )

        for t in clusters_words:
            print(t[:10])
        cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, clusters_words, cs_type='c_v')))
        print(cs_c_v)


if __name__ == "__main__":

    do_lemmatizing = True
    do_stop_word_removal = True

    data_processed, data_labels, vocab, tokenized_docs = preprocessing(
        new_data, new_data_label, do_lemmatizing=do_lemmatizing, do_stop_word_removal=do_stop_word_removal)

    test_data_processed, test_data_labels, test_vocab, test_tokenized_docs = preprocessing(
        new_test_data, new_test_data_label, do_lemmatizing=do_lemmatizing, do_stop_word_removal=do_stop_word_removal)

    #####
    # document space
    ####
    get_baseline(data_processed, vocab, tokenized_docs, data_labels, test_tokenized_docs)
    # not used: doc_clustering(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="w2v_avg")

    #####
    # word space
    ####
    # get_w2v_vis_sign_words(all_data_processed, vocab, tokenized_docs)
    # get_w2v_vis_topic_vec(all_data_processed, vocab, tokenized_docs)
    # get_graph_components(all_data_processed, vocab, tokenized_docs)
    # get_sage_graph_k_components(all_data, all_data_processed, vocab, tokenized_docs)
    # bert_visualization(all_data_processed, vocab, tokenized_docs)

    ####
    # word + doc space
    ####
    # w_d_clustering(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="doc2vec")
    # test_clustering(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="w2v_avg")

    # doc_clustering(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="w2v_avg")

    # bert_visualization(all_data_processed, vocab, tokenized_docs)
    # w_d_get_graph_components(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="w2v_avg")


    ####
    # Misc
    ####
    # number_of_words_per_doc()
    # vis_most_common_words(data_processed, raw_data=False, preprocessed=True)





    # new_vocab, vocab_embeddings = get_fast_text_embeddings(vocab)

    # w2v_visualization(all_data_processed, vocab, tokenized_docs)
    # get_w2v_vis_sign_words(all_data_processed, vocab, tokenized_docs)
    # doc2vec_visualization(all_data_processed, vocab, tokenized_docs)

    # fast_text_visualization(all_data_processed, vocab, tokenized_docs)

    # bert_visualization(all_data_processed, vocab, tokenized_docs)

    # graph_k_components(all_data, all_data_processed, tokenized_docs)

    # sage_graph_k_components(all_data, all_data_processed, vocab, tokenized_docs)

    # doc_clustering(all_data_processed, all_data_labels, vocab, tokenized_docs)

    # karate_club(all_data, all_data_processed, vocab, tokenized_docs)

    # alberta_visualization(all_data_processed, vocab, tokenized_docs)

    # get_w2v_vis_topic_vec(all_data_processed, vocab, tokenized_docs)



"""
def graph_k_components(original_data, all_data_processed, vocab, tokenized_docs):

    vocab_words, vocab_embeddings, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")

    best_c_v = {1: 0, 2: 0, 3: 0}
    best_c_v_topics = {1: None, 2: None, 3: None}

    worst_c_v = {1: 1, 2: 1, 3: 1}
    worst_c_v_topics = {1: None, 2: None, 3: None}

    y_c_v_clustering_type = {1: [], 2: [], 3: []}
    y_dbs_clustering_type = {1: [], 2: [], 3: []}
    y_u_mass_clustering_type = {1: [], 2: [], 3: []}

    x = [x/100 for x in range(40, 90, 10)]

    for sim in x:

        graph = create_networkx_graph(vocab_words, vocab_embeddings, similarity_threshold=sim)

        for k_component in [1, 2, 3]:

            cluster_words, _ = graph_evaluation_visualisation(
                graph,all_data_processed, vocab_words, k_component=k_component, word_rank_type="tf")

            if len(cluster_words) <= 2:
                cs_c_v = -1000.0
                dbs = -1000.0
                cs_npmi = -1000.0
            else:

                cluster_embeddings = [[w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)] for w in words]
                                  for words in cluster_words]

                cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, cluster_words, cs_type='c_v')))
                dbs = float("{:.2f}".format(davies_bouldin_index(cluster_embeddings)))
                cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed, cluster_words,
                                                                    len(cluster_words))))

            y_c_v_clustering_type[k_component].append(cs_c_v)
            y_u_mass_clustering_type[k_component].append(cs_npmi)
            y_dbs_clustering_type[k_component].append(dbs)

            if cs_c_v > best_c_v[k_component]:
                best_c_v[k_component] = cs_c_v
                best_c_v_topics[k_component] = cluster_words

            if cs_c_v < worst_c_v[k_component] and cs_c_v != -1000.0:
                worst_c_v[k_component] = cs_c_v
                worst_c_v_topics[k_component] = cluster_words





def sage_graph_k_components(original_data, all_data_processed, vocab, tokenized_docs):

    vocab_words, vocab_embeddings, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")

    best_c_v = {1: 0, 2: 0, 3: 0}
    best_c_v_topics = {1: None, 2: None, 3: None}

    worst_c_v = {1: 1, 2: 1, 3: 1}
    worst_c_v_topics = {1: None, 2: None, 3: None}

    y_c_v_clustering_type = {1: [], 2: [], 3: []}
    y_dbs_clustering_type = {1: [], 2: [], 3: []}
    y_u_mass_clustering_type = {1: [], 2: [], 3: []}

    x = [0.4, 0.5]

    for sim in x:

        graph = create_networkx_graph(vocab_words, vocab_embeddings, similarity_threshold=sim)

        for k_component in [1, 2, 3]:

            cluster_words, _ = graph_evaluation_visualisation(graph, all_data_processed, vocab_words,
                                                              k_component=k_component,
                                                              word_rank_type="tf")

            if len(cluster_words) <= 2:
                cs_c_v = -1000.0
                dbs = -1000.0
                cs_npmi = -1000.0

            else:

                node_sentence_embeddings, node_doc_embeddings = get_avg_sentence_doc_embeddings_w2v(original_data,
                                                                                                    list(graph.nodes()),
                                                                                                    vocab_words,
                                                                                                    vocab_embeddings)
                node_features = []
                for node in graph.nodes():
                    embedding = w2v_model.wv.vectors[w2v_model.wv.index2word.index(node)]
                    sentence_embedding = node_sentence_embeddings[node]
                    doc_embedding = node_doc_embeddings[node]

                    node_feature = embedding.tolist()
                    node_feature.extend(sentence_embedding)
                    node_feature.extend(doc_embedding)

                    node_features.append(node_feature)

                feature_graph = create_graph_with_features(graph, list(graph.nodes()), node_features)

                sg_graph = networkx_to_stellargraph(feature_graph)

                sg_words, sg_embeddings = graph_sage_embeddings(sg_graph)

                graph_revised = create_networkx_graph(sg_words, sg_embeddings, similarity_threshold=sim)

                cluster_words, _ = graph_evaluation_visualisation(graph=graph_revised, processed_data=all_data_processed,
                                                                  vocab=sg_words, k_component=k_component,
                                                                  word_rank_type="tf")

                if len(cluster_words) <= 2:
                    cs_c_v = -1000.0
                    dbs = -1000.0
                    cs_npmi = -1000.0
                else:

                    cluster_embeddings = [[w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)] for w in words]
                                          for words in cluster_words]

                    cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, cluster_words, cs_type='c_v')))
                    dbs = float("{:.2f}".format(davies_bouldin_index(cluster_embeddings)))
                    cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed, cluster_words,
                                                                        len(cluster_words))))

            y_c_v_clustering_type[k_component].append(cs_c_v)
            y_u_mass_clustering_type[k_component].append(cs_npmi)
            y_dbs_clustering_type[k_component].append(dbs)

            if cs_c_v > best_c_v[k_component]:
                best_c_v[k_component] = cs_c_v
                best_c_v_topics[k_component] = cluster_words

            if cs_c_v < worst_c_v[k_component] and cs_c_v != -1000.0:
                worst_c_v[k_component] = cs_c_v
                worst_c_v_topics[k_component] = cluster_words

    print("best c_v scores:")
    for k, b_cs in best_c_v.items():
        print(str(k) + ": " + str(b_cs))

    # c_v coherence score
    ys = [l for l in y_c_v_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="Coherence Score (c_v)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_v')
    fig.savefig("visuals/c_v_graph_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_u_mass_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="NPMI",
                          color_legends=["K=1", "K=2", "K=3"], type='c_npmi')
    fig.savefig("visuals/c_npmi_graph_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="Davies–Bouldin index",
                          color_legends=["K=1", "K=2", "K=3"], type='dbs')
    fig.savefig("visuals/dbi_graph_vs_k.pdf", bbox_inches='tight', transparent=True)

    best_c_v_topics_lengths = {1: None, 2: None, 3: None}
    for m, topics in best_c_v_topics.items():
        g, plt = create_circle_tree(topics)
        fig = plt.gcf()
        fig.savefig("visuals/best_" + str(m) + ".pdf", dpi=100, transparent=True)
        nx.write_graphml(g, "visuals/best_" + str(m) + ".graphml")

        # add to best_c_v_topics_lengths
        best_c_v_topics_lengths[m] = [len(t) for t in topics]

        # write topics
        write_topics_viz(topics, best_c_v[m], str(m),
                         "visuals/best_" + str(m) + ".txt")
        write_topics_viz(worst_c_v_topics[m], worst_c_v[m], str(m),
                         "visuals/worst_" + str(m) + ".txt")

    best_topics_lengths = [l for l in best_c_v_topics_lengths.values()]
    _, fig = box_plot(best_topics_lengths, ["K=1", "K=2", "K=3"], "k-Components", "Topic Lengths")
    fig.savefig("visuals/box_plot_graph.pdf", dpi=100, transparent=True)













    print("best c_v scores:")
    for k, b_cs in best_c_v.items():
        print(str(k) + ": " + str(b_cs))

    # c_v coherence score
    ys = [l for l in y_c_v_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="Coherence Score (c_v)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_v')
    fig.savefig("visuals/c_v_graph_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_u_mass_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="NPMI",
                          color_legends=["K=1", "K=2", "K=3"], type='c_npmi')
    fig.savefig("visuals/c_npmi_graph_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="Davies–Bouldin index",
                          color_legends=["K=1", "K=2", "K=3"], type='dbs')
    fig.savefig("visuals/dbi_graph_vs_k.pdf", bbox_inches='tight', transparent=True)

    best_c_v_topics_lengths = {1: None, 2: None, 3: None}
    for m, topics in best_c_v_topics.items():
        g, plt = create_circle_tree(topics)
        fig = plt.gcf()
        fig.savefig("visuals/best_" + str(m) + ".pdf", dpi=100, transparent=True)
        nx.write_graphml(g, "visuals/best_" + str(m) + ".graphml")

        # add to best_c_v_topics_lengths
        best_c_v_topics_lengths[m] = [len(t) for t in topics]

        # write topics
        write_topics_viz(topics, best_c_v[m], str(m),
                         "visuals/best_" + str(m) + ".txt")
        write_topics_viz(worst_c_v_topics[m], worst_c_v[m], str(m),
                         "visuals/worst_" + str(m) + ".txt")

    best_topics_lengths = [l for l in best_c_v_topics_lengths.values()]
    _, fig = box_plot(best_topics_lengths, ["K=1", "K=2", "K=3"], "k-Components", "Topic Lengths")
    fig.savefig("visuals/box_plot_graph.pdf", dpi=100, transparent=True)


"""