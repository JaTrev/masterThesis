from src.get_data import *
from src.preprocessing import *
from src.model import *
from src.evaluation import *
from src.clustering import *
from src.vectorization import *
from src.visualizations import *
from src.bert import *

import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import Counter

import networkx as nx

data, test_data = get_data()

new_data = [doc for doc in data if len([w for w in doc.split() if w.isalpha()]) > 2]
print("removed docs: " + str(len(data) - len(new_data)))
new_test_data = [doc for doc in test_data if len([w for w in doc.split() if w.isalpha]) > 2]
print("removed test docs: " + str(len(test_data) - len(new_test_data)))

all_data = [d for d in new_data]
all_data.extend(new_test_data)
# TODO: create a main() function


"""
topics = nmf_topics(all_data_processed, n_topics=10, solver='cd', beta_loss='frobenius', use_tfidf=False)

g, plt = create_circle_tree(topics)
nx.write_graphml(g, "test.graphml")
fig = plt.gcf()
fig.savefig('visuals/test.pdf', dpi=100, transparent=True)

assert 0
"""
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


def vis_most_common_words(corpus = None):
    if corpus == None:
        corpus = [doc.split() for doc in all_data]

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
    for doc in corpus:
        data_words.extend([w.lower() for w in doc if w.isalpha()])

    data_words_c = Counter(data_words)

    most_common_words = [w for w, c in data_words_c.most_common(30)]
    most_common_words_c = [c for w, c in data_words_c.most_common(30)]

    plt.bar(most_common_words, most_common_words_c, color='black', width=0.5)

    plt.ylim(top=1000) # most_common_words_c[0])
    plt.ylim(bottom=0)

    ax.set_xlabel("Top 30 Words", fontsize="medium")
    ax.set_ylabel("Number of Occurrences", fontsize="medium")

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    fig.savefig("visuals/processed_most_common_words.pdf", bbox_inches='tight', transparent=True)

def get_baseline_vis(all_data_processed: list):
    x = list(range(2, 22, 2))
    y_nmf = []
    best_nmf_cs = 0
    best_nmf_topics = None

    y_lsa = []
    best_lsa_cs = 0
    best_lsa_topics = None

    y_lda = []
    best_lda_cs = 0
    best_lda_topics = None

    y_lda_mallet = []
    best_lda_mallet_cs = 0
    best_lda_mallet_topics = None

    for n_topic in x:
        topics = nmf_topics(all_data_processed, n_topics=n_topic, solver='cd', beta_loss='frobenius', use_tfidf=False)
        cs = coherence_score(all_data_processed, topics)
        y_nmf.append(cs)
        if cs > best_nmf_cs:
            best_nmf_cs = cs
            best_nmf_topics = topics

        topics = lsa_topics(all_data_processed, n_topics=n_topic)
        cs = coherence_score(all_data_processed, topics)
        y_lsa.append(cs)
        if cs > best_lsa_cs:
            best_lsa_cs = cs
            best_lsa_topics = topics

        topics = lda_topics(all_data_processed, n_topics=n_topic)
        cs = coherence_score(all_data_processed, topics)
        y_lda.append(cs)
        if cs > best_lda_cs:
            best_lda_cs = cs
            best_lda_topics = topics

        topics = lda_mallet_topics(all_data_processed, n_topics=n_topic)
        cs = coherence_score(all_data_processed, topics)
        y_lda_mallet.append(cs)
        if cs > best_lda_mallet_cs:
            best_lda_mallet_cs = cs
            best_lda_mallet_topics = topics

    ys = [y_nmf, y_lsa, y_lda, y_lda_mallet]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score",
                          color_legends=["NMF", "LSA", "LDA", "LDA MALLET"])

    fig.savefig("visuals/cs_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)

    g, plt = create_circle_tree(best_nmf_topics)
    fig = plt.gcf()
    fig.savefig('visuals/best_nmf.pdf', dpi=100, transparent=True)
    nx.write_graphml(g, "visuals/best_nmf.graphml")

    g, plt = create_circle_tree(best_lsa_topics)
    fig = plt.gcf()
    fig.savefig('visuals/best_lsa.pdf', dpi=100, transparent=True)
    nx.write_graphml(g, "visuals/best_lsa.graphml")

    g, plt = create_circle_tree(best_lda_topics)
    fig = plt.gcf()
    fig.savefig('visuals/best_lda.pdf', dpi=100, transparent=True)
    nx.write_graphml(g, "visuals/best_lda.graphml")

    g, plt = create_circle_tree(best_lda_mallet_topics)
    fig = plt.gcf()
    fig.savefig('visuals/best_lda_mallet.pdf', dpi=100, transparent=True)
    nx.write_graphml(g, "visuals/best_lda_mallet.graphml")

def w2v_visualizsation(all_data_processed, vocab):
    words, word_embeddings, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")

    word_similarities = get_word_similarities(word_embeddings)

    # 'kmeans', 'agglomerative', 'spectral'
    x = list(range(2, 22, 2))
    y_kmeans_cs = []
    y_kmeans_dbs = []
    best_kmeans_cs = 0
    best_kmeans_topics = None

    y_agglo_cs = []
    y_agglo_dbs = []
    best_agglo_cs = 0
    best_agglo_topics = None

    y_kmeans_sim_cs = []
    y_kmeans_sim_dbs = []
    best_kmeans_sim_cs = 0
    best_kmeans_sim_topics = None

    y_agglo_sim_cs = []
    y_agglo_sim_dbs = []
    best_agglo_sim_cs = 0
    best_agglo_sim_topics = None

    for k in x:
        # kmeans
        clusters_words, clusters_words_embeddings = word_clusters(all_data_processed, words, word_embeddings, vocab,
                                                                  clustering_type="kmeans",
                                                                  params={'n_clusters': k, 'random_state': 42, })

        cs = float("{:.2f}".format(coherence_score(all_data_processed, clusters_words, cs_type='c_v')))
        dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
        y_kmeans_cs.append(cs)
        y_kmeans_dbs.append(dbs)
        if best_kmeans_cs < cs or best_kmeans_cs == 0:
            best_kmeans_cs = cs
            best_kmeans_topics = clusters_words

        # agglomerative
        clusters_words, clusters_words_embeddings = word_clusters(all_data_processed, words, word_embeddings, vocab,
                                                                  clustering_type="agglomerative",
                                                                  params={'n_clusters': k, })
        cs = float("{:.2f}".format(coherence_score(all_data_processed, clusters_words, cs_type='c_v')))
        dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
        y_agglo_cs.append(cs)
        y_agglo_dbs.append(dbs)

        if best_agglo_cs < cs or best_agglo_cs == 0:
            best_agglo_cs = cs
            best_agglo_topics = clusters_words

        # kmeans similarity embeddings
        clusters_words, clusters_words_embeddings = word_clusters(all_data_processed, words, word_similarities, vocab,
                                                                  clustering_type="kmeans",
                                                                  params={'n_clusters': k, 'random_state': 42, })
        cs = float("{:.2f}".format(coherence_score(all_data_processed, clusters_words, cs_type='c_v')))
        dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
        y_kmeans_sim_cs.append(cs)
        y_kmeans_sim_dbs.append(dbs)

        if best_kmeans_sim_cs < cs or best_kmeans_sim_cs == 0:
            best_kmeans_sim_cs = cs
            best_kmeans_sim_topics = clusters_words

        # agglomerative similarities embedding
        clusters_words, clusters_words_embeddings = word_clusters(all_data_processed, words, word_similarities, vocab,
                                                                  clustering_type="agglomerative",
                                                                  params={'n_clusters': k, })
        cs = float("{:.2f}".format(coherence_score(all_data_processed, clusters_words, cs_type='c_v')))
        dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
        y_agglo_sim_cs.append(cs)
        y_agglo_sim_dbs.append(dbs)

        if best_agglo_sim_cs < cs or best_agglo_sim_cs == 0:
            best_agglo_sim_cs = cs
            best_agglo_sim_topics = clusters_words

    # w2v, kmeans
    g, plt = create_circle_tree([c[:10] for c in best_kmeans_topics])
    fig = plt.gcf()
    fig.savefig("visuals/w2v_kmeans_cs" + str(best_kmeans_cs) + ".pdf", dpi=100, transparent=True)
    nx.write_graphml(g, "visuals/w2v_kmeans_cs" + str(best_kmeans_cs) + ".graphml")

    # w2v, agglomerative
    g, plt = create_circle_tree([c[:10] for c in best_agglo_topics])
    fig = plt.gcf()
    fig.savefig("visuals/w2v_agglo_cs" + str(best_agglo_cs) + ".pdf", dpi=100, transparent=True)
    nx.write_graphml(g, "visuals/w2v_agglo_cs" + str(best_agglo_cs) + ".graphml")

    # similarities, kmeans
    g, plt = create_circle_tree([c[:10] for c in best_kmeans_sim_topics])
    fig = plt.gcf()
    fig.savefig("visuals/w2v_kmeans_sim_cs" + str(best_kmeans_sim_cs) + ".pdf", dpi=100, transparent=True)
    nx.write_graphml(g, "visuals/w2v_kmeans_sim_cs" + str(best_kmeans_sim_cs) + ".graphml")

    # similarities, agglomerative
    g, plt = create_circle_tree([c[:10] for c in best_agglo_sim_topics])
    fig = plt.gcf()
    fig.savefig("visuals/w2v_agglo_sim_cs" + str(best_agglo_sim_cs) + ".pdf", dpi=100, transparent=True)
    nx.write_graphml(g, "visuals/w2v_agglo_sim_cs" + str(best_agglo_sim_cs) + ".graphml")

    # box plot
    best_topics_lengths = []
    best_topics_lengths.append([len(t) for t in best_kmeans_topics])
    best_topics_lengths.append([len(t) for t in best_agglo_topics])
    best_topics_lengths.append([len(t) for t in best_kmeans_sim_topics])
    best_topics_lengths.append([len(t) for t in best_agglo_sim_topics])
    _, fig = box_plot(best_topics_lengths, ["K-Means", "Agglomerative", "K-Means+S", "Agglomerative+S"],
                      "Clustering Types", "Topic Lengths")
    fig.savefig("visuals/box_plot_w2v_sim.pdf", dpi=100, transparent=True)

    ys = [y_kmeans_cs, y_agglo_cs, y_kmeans_sim_cs, y_agglo_sim_cs]
    print(ys)
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score", type="c_v",
                          color_legends=["K-Means", "Agglomerative",
                                         "K-Means + Sim", "Agglomerative + Sim"])
    fig.savefig("visuals/scatter_plot_w2v_sim_cs.pdf", dpi=100, transparent=True)

    ys = [y_kmeans_dbs, y_agglo_dbs, y_kmeans_sim_dbs, y_agglo_sim_dbs]
    print(ys)
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Davies-Bouldin Score", type="dbs",
                          color_legends=["K-Means", "Agglomerative",
                                         "K-Means + Sim", "Agglomerative + Sim"])
    fig.savefig("visuals/scatter_plot_w2v_sim_dbs.pdf", dpi=100, transparent=True)


if __name__ == "__main__":
    # all_data_lengths = [len([w for w in doc.split() if w.isalpha()]) for doc in new_data]
    all_data_processed, vocab = preprocessing(all_data, do_stemming=True)

