from src.get_data import *
from src.preprocessing import *
from src.model import *
from src.evaluation import *
from src.clustering import *
from src.vectorization import *
from src.visualizations import *
from src.bert import *
from src.graphs import *

import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import Counter
import pickle
import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score
from karateclub import DeepWalk, GEMSEC

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


all_data = [d for d in new_data]
all_data.extend(new_test_data)

all_data_label = [l for l in new_data_label]
all_data_label.extend(new_test_data_label)

assert len(all_data) == len(all_data_label)


# TODO: create a main() function


def number_of_words_per_doc():
    fig, ax = plt.subplots(figsize=(10, 6))

    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 2

    ax.tick_params(axis='both', labelsize=18)

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
    ax.set_ylabel("Number of Documents", fontsize="medium")

    plt.show()
    fig.savefig("visuals/document_word_distribution.pdf", bbox_inches='tight', transparent=True)


def vis_most_common_words(data: list):
    # if corpus == None:
    #    corpus = [doc.split() for doc in all_data]

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

    plt.ylim(top=1000)
    plt.ylim(bottom=0)

    ax.set_xlabel("Top 30 Words", fontsize="medium")
    ax.set_ylabel("Number of Occurrences", fontsize="medium")

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    fig.savefig("visuals/processed_most_common_words.pdf", bbox_inches='tight', transparent=True)


def get_baseline_vis(all_data_processed: list, vocab: list, x: list = None):
    if x is None:
        x = list(range(2, 22, 2))
    else:
        assert isinstance(x, list), "x has to be a list of ks"

    k_10_c_v = {'nmf_tf': 0, 'nmf_tf_idf': 0, 'lda': 0, 'lda_mallet': 0}
    best_c_v = {'nmf_tf': 0, 'nmf_tf_idf': 0, 'lda': 0, 'lda_mallet': 0}
    k_10_topics = {'nmf_tf': None, 'nmf_tf_idf': None, 'lda': None, 'lda_mallet': None}
    best_c_v_topics = {'nmf_tf': None, 'nmf_tf_idf': None, 'lda': None, 'lda_mallet': None}

    y_c_v_models = {'nmf_tf': [], 'nmf_tf_idf': [], 'lda': [], 'lda_mallet': []}
    y_u_mass_models = {'nmf_tf': [], 'nmf_tf_idf': [], 'lda': [], 'lda_mallet': []}

    for n_topic in x:

        for m in list(best_c_v.keys()):

            if m == 'nmf_tf':
                topics = nmf_topics(all_data_processed, vocabulary=vocab, n_topics=n_topic, solver='cd',
                                    beta_loss='frobenius', use_tfidf=False)
            elif m == 'nmf_tf_idf':
                topics = nmf_topics(all_data_processed, vocabulary=vocab, n_topics=n_topic, solver='cd',
                                    beta_loss='frobenius', use_tfidf=True)
            elif m == 'lda':
                topics = lda_topics(all_data_processed, n_topics=n_topic)

            elif m == 'lda_mallet':
                topics = lda_mallet_topics(all_data_processed, n_topics=n_topic)

            else:
                topics = []
                assert m in best_c_v.keys()

            # c_v coherence score
            cs_c_v = coherence_score(all_data_processed, topics, cs_type='c_v')
            y_c_v_models[m].append(cs_c_v)
            if cs_c_v > best_c_v[m]:
                best_c_v[m] = cs_c_v
                best_c_v_topics[m] = topics

            # u_mass coherence score
            cs_u_mass = coherence_score(all_data_processed, topics, cs_type='u_mass')
            y_u_mass_models[m].append(cs_u_mass)

            if n_topic == 10:
                k_10_c_v[m] = cs_c_v
                k_10_topics[m] = topics

    print("best c_v scores:")
    for m, b_cs in best_c_v.items():
        print(str(m) + ": " + str(b_cs))

    print()
    print("k=10 c_v scores:")
    for m, cs_score in k_10_c_v.items():
        print(str(m) + ": " + str(cs_score))

    # c_v coherence score
    ys = [l for l in y_c_v_models.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["NMF TF", "NMF TF-IDF", "LDA", "LDA MALLET"], type='c_v')
    fig.savefig("visuals/c_v_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_u_mass_models.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (u_mass)",
                          color_legends=["NMF TF", "NMF TF-IDF", "LDA", "LDA MALLET"], type='u_mass')
    fig.savefig("visuals/u_mass_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)

    # best_c_v_topics_lengths = {'nmf_tf': None, 'nmf_tf_idf': None, 'lda': None, 'lda_mallet': None}
    for m, topics in best_c_v_topics.items():
        g, plt = create_circle_tree(topics)
        fig = plt.gcf()
        fig.savefig("visuals/best_" + str(m) + ".pdf", dpi=100, transparent=True)
        nx.write_graphml(g, "visuals/best_" + str(m) + ".graphml")

        # k = 10 model
        g, plt = create_circle_tree(k_10_topics[m])
        nx.write_graphml(g, "visuals/k=10_" + str(m) + ".graphml")

        # add to best_c_v_topics_lengths
        # best_c_v_topics_lengths[m] = [len(t) for t in topics]

        # write topics
        write_topics(topics, "visuals/best_" + str(m) + ".txt")
        # write k = 10 model
        write_topics(k_10_topics[m], "visuals/k=10_" + str(m) + ".txt")


def w2v_visualization(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):
    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    _, _, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")
    print("min_count: " + str(w2v_model.min_count))
    print("window: " + str(w2v_model.window))
    print("negative: " + str(w2v_model.negative))
    print("ns_exponent: " + str(w2v_model.ns_exponent))
    w2v_params = {'min_c': w2v_model.min_count, 'win': w2v_model.window, 'negative': w2v_model.negative,
                  'ns_exponent': w2v_model.ns_exponent, 'seed': 42, 'sample': w2v_model.sample}

    words, word_embeddings, _, _ = get_doc2vec_embeddings(all_data_processed, vocab, **w2v_params)

    # get_word_vectors(all_data_processed, vocab, params=w2v_params)
    # words, word_embeddings = get_glove_embeddings(vocab)

    # get word embedding similarities
    # word_embeddings = get_word_similarities(word_embeddings)

    k_10_c_v = {"kmeans": 0, "agglomerative": 0, "spectral": 0, "nmf": 0}
    best_c_v = {"kmeans": 0, "agglomerative": 0, "spectral": 0, "nmf": 0}
    k_10_topics = {"kmeans": None, "agglomerative": None, "spectral": None, "nmf": None}
    best_c_v_topics = {"kmeans": None, "agglomerative": None, "spectral": None, "nmf": None}

    worst_c_v = {"kmeans": 1, "agglomerative": 1, "spectral": 1, "nmf": 1}
    worst_c_v_topics = {"kmeans": None, "agglomerative": None, "spectral": None, "nmf": None}

    y_c_v_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": [], "nmf": []}
    y_dbs_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": [], "nmf": []}
    y_u_mass_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": [], "nmf": []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "spectral", "nmf"]:

            if cluster_type in ["kmeans", "nmf"]:
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
            # cs_u_muss = float("{:.2f}".format(coherence_score(all_data_processed, clusters_words, cs_type='u_mass')))
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
                          color_legends=["K-Means", "Agglomerative", "Spectral", "NMF"], type='c_v')
    fig.savefig("visuals/c_v_w2v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_u_mass_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="NPMI",
                          color_legends=["K-Means", "Agglomerative", "Spectral", "NMF"], type='c_npmi')
    fig.savefig("visuals/c_npmi_w2v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative", "Spectral", "NMF"], type='dbs')
    fig.savefig("visuals/dbi_w2v_vs_k.pdf", bbox_inches='tight', transparent=True)

    best_c_v_topics_lengths = {"kmeans": None, "agglomerative": None, "spectral": None, "nmf": None}
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
    _, fig = box_plot(best_topics_lengths, ["K-Means", "Agglomerative", "Spectral", "NMF"], "Clustering Types",
                      "Topic Lengths")
    fig.savefig("visuals/box_plot_w2v.pdf", dpi=100, transparent=True)


def w2v_ablation(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):
    x_label = "Minimum Sampling"
    file_save_under = "min"
    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 11, 1))

    else:
        assert isinstance(x, list)

    _, _, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")
    # w2v_model
    # min_count: 50
    # window: 3
    # negative: 60
    # ns_exponent: 0.75
    orig = w2v_model.min_count
    print(orig)

    best_var = {"kmeans": 0, "agglomerative": 0, "spectral": 0}
    best_c_v_topics = {"kmeans": None, "agglomerative": None, "spectral": None}

    worst_var = {"kmeans": 1, "agglomerative": 1, "spectral": 1}
    worst_c_v_topics = {"kmeans": None, "agglomerative": None, "spectral": None}

    y_c_v_var = {"kmeans": [], "agglomerative": [], "spectral": []}
    y_dbs_var = {"kmeans": [], "agglomerative": [], "spectral": []}
    y_u_mass_var = {"kmeans": [], "agglomerative": [], "spectral": []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "spectral"]:

            w2v_params = {'min_c': k, 'win': w2v_model.window, 'negative': w2v_model.negative,
                          'ns_exponent': w2v_model.ns_exponent, 'seed': 42}
            words, word_embeddings, _ = get_word_vectors(all_data_processed, vocab, params=w2v_params)

            if cluster_type == "kmeans":
                clustering_params = {'n_clusters': 10, 'random_state': 42, }
            else:
                clustering_params = {'n_clusters': 10}

            clusters_words, clusters_words_embeddings = word_clusters(
                all_data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                params=clustering_params, clustering_weight_type=clustering_weight_type,
                ranking_weight_type=ranking_weight_type
            )

            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, clusters_words, cs_type='c_v')))
            dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
            # cs_u_mass = float("{:.2f}".format(coherence_score(tokenized_docs, clusters_words, cs_type='u_mass')))
            cs_npmi = average_npmi_topics(all_data_processed, clusters_words, len(clusters_words))

            y_c_v_var[cluster_type].append(cs_c_v)
            y_u_mass_var[cluster_type].append(cs_npmi)
            y_dbs_var[cluster_type].append(dbs)

            if cs_c_v > best_var[cluster_type]:
                best_var[cluster_type] = cs_c_v
                best_c_v_topics[cluster_type] = clusters_words

            if cs_c_v < worst_var[cluster_type]:
                worst_var[cluster_type] = cs_c_v
                worst_c_v_topics[cluster_type] = clusters_words

            if k == orig:
                print(str(cluster_type) + ": " + str(cs_c_v))

    print("best c_v scores:")
    for m, b_cs in best_var.items():
        print(str(m) + ": " + str(b_cs))

    # c_v coherence score
    ys = [l for l in y_c_v_var.values()]
    _, fig = scatter_plot(x, ys, x_label=x_label, y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='c_v')
    fig.savefig("visuals/c_v_w2v_vs_" + str(file_save_under) + ".pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_u_mass_var.values()]
    _, fig = scatter_plot(x, ys, x_label=x_label, y_label="NPMI",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='c_npmi')
    fig.savefig("visuals/npmi_w2v_vs_" + str(file_save_under) + ".pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_var.values()]
    _, fig = scatter_plot(x, ys, x_label=x_label, y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='dbs')
    fig.savefig("visuals/dbi_w2v_vs_" + str(file_save_under) + ".pdf", bbox_inches='tight', transparent=True)

    best_c_v_topics_lengths = {"kmeans": None, "agglomerative": None, "spectral": None}
    for m, topics in best_c_v_topics.items():
        g, plt = create_circle_tree(topics)
        fig = plt.gcf()
        fig.savefig("visuals/best_" + str(m) + "_" + str(file_save_under) + ".pdf", dpi=100, transparent=True)
        nx.write_graphml(g, "visuals/best_" + str(m) + "_" + str(file_save_under) + ".graphml")

        # add to best_c_v_topics_lengths
        best_c_v_topics_lengths[m] = [len(t) for t in topics]

        # write topics
        write_topics_ablation(topics, best_var[m], "visuals/best_" + str(m) + "_" + str(file_save_under) + ".txt")
        write_topics_ablation(topics, worst_var[m],
                              "visuals/worst_" + str(m) + "_" + str(file_save_under) + ".txt")

    best_topics_lengths = [l for l in best_c_v_topics_lengths.values()]
    _, fig = box_plot(best_topics_lengths, ["K-Means", "Agglomerative", "Spectral"], "Clustering Methods",
                      "Topic Lengths")
    fig.savefig("visuals/box_plot_w2v_" + str(file_save_under) + ".pdf", dpi=100, transparent=True)


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


def bert_visualization(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):

    with open("data/all_vocab_emb_dict_last_bare.pickle", "rb") as f:
        all_vocab_emb_dict = pickle.load(f)

    all_vocab_emb_dict_words = all_vocab_emb_dict.keys()
    vocab_bert_embeddings = []
    new_vocab = []
    for word in vocab:

        if word in all_vocab_emb_dict_words:

            w_embeddings = all_vocab_emb_dict[word]

            if len(w_embeddings) >= 100:
                new_vocab.append(word)
                vocab_bert_embeddings.append(np.average(w_embeddings, axis=0))
            else:
                continue
    assert all(len(embd) == 768 for embd in vocab_bert_embeddings)

    print("old vocab len: " + str(len(vocab)))
    print("new vocab len: " + str(len(new_vocab)))

    vocab = new_vocab
    vocab_embeddings = vocab_bert_embeddings

    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    words, word_embeddings = (vocab, vocab_embeddings)

    k_10_c_v = {"kmeans": 0, "agglomerative": 0, "nmf": 0}
    best_c_v = {"kmeans": 0, "agglomerative": 0, "nmf": 0}
    k_10_topics = {"kmeans": None, "agglomerative": None, "nmf": None}
    best_c_v_topics = {"kmeans": None, "agglomerative": None, "nmf": None}

    worst_c_v = {"kmeans": 1, "agglomerative": 1, "nmf": 1}
    worst_c_v_topics = {"kmeans": None, "agglomerative": None, "nmf": None}

    y_c_v_clustering_type = {"kmeans": [], "agglomerative": [], "nmf": []}
    y_dbs_clustering_type = {"kmeans": [], "agglomerative": [], "nmf": []}
    y_u_mass_clustering_type = {"kmeans": [], "agglomerative": [], "nmf": []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "nmf"]:

            if cluster_type in ["kmeans", "nmf"]:
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
            cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed,
                                                                clusters_words, len(clusters_words))))

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
                          color_legends=["K-Means", "Agglomerative", "NMF"], type='c_v')
    fig.savefig("visuals/c_v_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_u_mass_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="NPMI",
                          color_legends=["K-Means", "Agglomerative", "NMF"], type='c_npmi')
    fig.savefig("visuals/c_npmi_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative", "NMF"], type='dbs')
    fig.savefig("visuals/dbi_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

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
    _, fig = box_plot(best_topics_lengths, ["K-Means", "Agglomerative", "NMF"], "Clustering Types", "Topic Lengths")
    fig.savefig("visuals/box_plot_bert.pdf", dpi=100, transparent=True)


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


def doc_clustering(all_data_processed: list, all_data_labels: list, vocab: list, tokenized_docs: list,
                   doc_embedding_type="w2v_mean", x: list = None):

    # main extrinsic evaluation metric: ARI
    # https://stats.stackexchange.com/questions/381223/evaluation-of-clustering-method

    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    _, _, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")
    print("min_count: " + str(w2v_model.min_count))
    print("window: " + str(w2v_model.window))
    print("negative: " + str(w2v_model.negative))
    print("ns_exponent: " + str(w2v_model.ns_exponent))
    w2v_params = {'min_c': w2v_model.min_count, 'win': w2v_model.window, 'negative': w2v_model.negative,
                  'ns_exponent': w2v_model.ns_exponent, 'seed': 42, 'sample': w2v_model.sample}

    doc_data, doc_labels, doc_embeddings, vocab = get_doc_embeddings(all_data_processed, all_data_labels,
                                                                     vocab, doc_embedding_type, w2v_params)

    k_10_ari = {"kmeans": 0, "agglomerative": 0, "spectral": 0}
    best_ari = {"kmeans": 0, "agglomerative": 0, "spectral": 0}
    k_10_topics = {"kmeans": None, "agglomerative": None, "spectral": None}
    best_ari_topics = {"kmeans": None, "agglomerative": None, "spectral": None}

    worst_ari = {"kmeans": 1, "agglomerative": 1, "spectral": 1}
    worst_ari_topics = {"kmeans": None, "agglomerative": None, "spectral": None}

    y_nmi_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": []}
    y_ari_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": []}
    y_acc_clustering_type = {"kmeans": [], "agglomerative": [], "spectral": []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "spectral"]:

            if cluster_type in ["kmeans", "nmf"]:
                clustering_params = {'n_clusters': k, 'random_state': 42, }
            else:
                clustering_params = {'n_clusters': k}

            clusters_docs, clusters_docs_embeddings, labels_predict = document_clustering(doc_data, doc_embeddings,
                                                                                          vocab, cluster_type,
                                                                                          params=clustering_params)

            ami = float("{:.2f}".format(ami_score(labels_true=doc_labels, labels_pred=labels_predict,
                                                  average_method='arithmetic')))
            ari = float("{:.2f}".format(ari_score(labels_true=doc_labels, labels_pred=labels_predict)))
            acc = float("{:.2f}".format(acc_score(labels_true=doc_labels, labels_pred=labels_predict,
                                                  sample_weight=None)))

            y_nmi_clustering_type[cluster_type].append(ami)
            y_acc_clustering_type[cluster_type].append(acc)
            y_ari_clustering_type[cluster_type].append(ari)

            if ari > best_ari[cluster_type]:
                best_ari[cluster_type] = ari
                best_ari_topics[cluster_type] = clusters_docs

            if ari < worst_ari[cluster_type]:
                worst_ari[cluster_type] = ari
                worst_ari_topics[cluster_type] = clusters_docs

            if k == 10:
                k_10_ari[cluster_type] = ari
                k_10_topics[cluster_type] = clusters_docs

    print("best ari scores:")
    for m, b_cs in best_ari.items():
        print(str(m) + ": " + str(b_cs))

    print()
    print("k=10 ari scores:")
    for m, cs_score in k_10_ari.items():
        print(str(m) + ": " + str(cs_score))

    # c_v coherence score
    ys = [l for l in y_nmi_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="AMI",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='ami')
    fig.savefig("visuals/ami_doc_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_acc_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="ACC",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='acc')
    fig.savefig("visuals/acc_doc_vs_k.pdf", bbox_inches='tight', transparent=True)

    # ari score
    ys = [l for l in y_ari_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="ARI",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='ari')
    fig.savefig("visuals/ari_doc_vs_k.pdf", bbox_inches='tight', transparent=True)


if __name__ == "__main__":
    all_data_processed, all_data_labels, vocab, tokenized_docs = preprocessing(all_data,
                                                                               all_data_label,
                                                                               do_stemming=False,
                                                                               do_lemmatizing=True,
                                                                               remove_low_freq=False)
    #
    # get_baseline_vis(all_data_processed, vocab)
    # vis_most_common_words(all_data_processed)
    # print("Trying Lemmatizing")

    # new_vocab, vocab_embeddings = get_fast_text_embeddings(vocab)

    # w2v_visualization(all_data_processed, vocab, tokenized_docs)
    # w2v_ablation(all_data_processed, vocab, tokenized_docs)

    # fast_text_visualization(all_data_processed, vocab, tokenized_docs)

    # bert_visualization(all_data_processed, vocab, tokenized_docs)

    # graph_k_components(all_data, all_data_processed, tokenized_docs)

    # sage_graph_k_components(all_data, all_data_processed, vocab, tokenized_docs)

    doc_clustering(all_data_processed, all_data_labels, vocab, tokenized_docs)
