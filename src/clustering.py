from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from src.misc import *
import numpy as np
from sklearn.decomposition import NMF
import hdbscan
import umap


def nmf_clustering(word_embeddings: list, words: list, n_clusters: int = 10, n_words: int = 10, random_state: int = 42,
                    init: str = 'nndsvd', solver: str = 'cd', beta_loss: str = 'frobenius'):

    """
    nmf_clustering uses the similarity matrix based on the word_embeddings to perform NMF factorization.
    The similarity matrix is pruned by removing small values which gives a sparser similarity space to operate in.

    based on "Rethinking Topic Modelling: From Document-Space to Term-Space" by Magnus Sahlgren
    :param word_embeddings:
    :param words:
    :param n_clusters:
    :param n_words:
    :param random_state:
    :param init:
    :param solver:
    :param beta_loss:
    :return:
    """

    assert len(word_embeddings) == len(words)
    assert init in [None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar'], "need an appropriate method to init to procedure"
    assert solver in ['mu', 'cd'], "need an appropriate solver"
    assert beta_loss in ['frobenius', 'kullback-leibler', 'itakura-saito'], "need an appropriate beta_loss"

    embeddings_similarities = []
    all_sim = []
    for i_w, w_embedding in enumerate(word_embeddings):
        w_similarities = cosine_similarity(w_embedding.reshape(1, -1),  word_embeddings)[0]
        w_similarities = [0 if sim <= 0 or i_sim == i_w else sim for i_sim, sim in enumerate(w_similarities)]
        all_sim.extend(w_similarities)
        embeddings_similarities.append(w_similarities)

    nmf_model = NMF(n_components=n_clusters, init=init, beta_loss=beta_loss, solver=solver,
                    max_iter=1000, alpha=.1, l1_ratio=.5, random_state=random_state).fit(embeddings_similarities)

    topics_words = []
    topics_embeddings = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_words = [words[i] for i in topic.argsort()[:-n_words - 1:-1]]
        top_embeddings = [word_embeddings[i] for i in topic.argsort()[:-n_words - 1:-1]]

        topics_words.append(top_words)
        topics_embeddings.append(top_embeddings)

    return topics_words, topics_embeddings


def kmeans_clustering(word_embeddings: list, word_weights: list = None, params: dict = None) -> list:
    model = KMeans(**params)
    return model.fit_predict(word_embeddings, word_weights)


def hdbscan_clustering(embeddings: list, min_cluster_size: int = 10, n_neighbors: int = 15, n_components: int = 5,
                       do_dim_reduction=False):

    if do_dim_reduction:
        umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                               metric='cosine', random_state=123).fit(embeddings)
        embeddings = umap_model.embedding_

    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                              cluster_selection_method='eom').fit(embeddings)

    return cluster.labels_, cluster.probabilities_


def agglomerative_clustering(word_embeddings: list, word_weights: list = None, params: dict = None):
    model = AgglomerativeClustering(**params)
    return model.fit_predict(word_embeddings, word_weights)


def spectral_clustering(word_embeddings: list, word_weights: list = None, params: dict = None):
    model = SpectralClustering(**params)
    return model.fit_predict(word_embeddings, word_weights)


def dbscan_cluster(word_embeddings: list, word_weights: list = None):
    return DBSCAN(min_samples=6).fit_predict(word_embeddings, sample_weight=word_weights)


def sort_words(processed_docs: list, cluster_words: list, cluster_embeddings: list,
               weight_type: str = "tf") -> (list, list):

    assert len(cluster_words) == len(cluster_embeddings), "cluster_words and cluster_embeddings do not " \
                                                          "have the same amount of clusters"
    assert all([len(cluster_words[i]) == len(cluster_embeddings[i]) for i in range(len(cluster_words))]), (
        "each cluster must have the same number of words and embeddings")

    assert weight_type in ["tf", "tf-df", "tf-idf"], "wrong counter_type!"

    # word_weight = None
    n_words = len([w for doc in processed_docs for w in doc])
    clusters_vocab = list(set([w for c_words in cluster_words for w in c_words]))

    word_weights = get_word_weights(processed_docs, vocab=clusters_vocab, n_words=n_words, weight_type=weight_type)

    # calculate cluster centers
    cluster_centers = [np.mean(c_embeddings, axis=0) for c_embeddings in cluster_embeddings]
    assert len(cluster_centers) == len(cluster_embeddings)
    assert len(cluster_centers[0]) == len(cluster_embeddings[0][0])

    # calculate cosine similarity to cluster center
    cluster_similarities = [cosine_similarity(cluster_embeddings[i_c],
                                              cluster_center.reshape(1, -1))
                            for i_c, cluster_center in enumerate(cluster_centers)]

    assert len(cluster_similarities) == len(cluster_embeddings)
    assert len(cluster_similarities[0]) == len(cluster_embeddings[0])

    # sort cluster_words
    sorted_cluster_words = []
    sorted_cluster_embeddings = []
    sorted_cluster_idxs = [sorted(range(len(c)),
                                  key=lambda k: word_weights[c[k]] * cluster_similarities[i_c][k],
                                  reverse=True)
                           for i_c, c in enumerate(cluster_words)]

    for i_c in range(len(cluster_words)):
        sorted_cluster_words.append([cluster_words[i_c][i] for i in sorted_cluster_idxs[i_c]])
        sorted_cluster_embeddings.append([cluster_embeddings[i_c][i] for i in sorted_cluster_idxs[i_c]])

    return sorted_cluster_words, sorted_cluster_embeddings


def word_clusters(processed_docs: list, words: list, word_embeddings: list, vocab: list,
                  clustering_type: str, params: dict,
                  clustering_weight_type: str = 'tf',
                  ranking_weight_type=None) -> (list, list):
    """
    word_clusters returns a sorted list of words for each cluster

    :param processed_docs: list of preprocessed documents
    :param words: list of words
    :param word_embeddings: list of word embeddings
    :param vocab: list of vocabulary words
    :param clustering_type: defines the clustering method ('kmeans', 'agglomerative', 'spectral')
    :param params: clustering parameters
    :param clustering_weight_type: word weighting type used for clustering ("tf", "tf-df", "tf-idf")
    :param ranking_weight_type: word weighting type used for ranking words ("tf", "tf-df", "tf-idf")

    :return: list of cluster words for each cluster, list of word embeddings for each cluster (sorted!)
    """
    # :param n_words: number of words for every cluster

    assert len(word_embeddings) == len(words), "word_embeddings and word list do not have the same length"
    assert clustering_type in ['kmeans', 'agglomerative', 'spectral', 'nmf'], "incorrect clustering_type"
    assert all([w in vocab for w in words]), "some words are not in the vocabulary"

    clustering_dict = {
        'kmeans': kmeans_clustering,
        'agglomerative': agglomerative_clustering,
        'spectral': spectral_clustering
    }

    if clustering_weight_type is None:
        print("Performing clustering without any weights!")

        word_weights = None

    else:
        print("Performing weighted clustering!")

        n_words = len([w for doc in processed_docs for w in doc])
        word_weights_dict = get_word_weights(processed_docs, vocab, n_words, weight_type=clustering_weight_type)
        word_weights = [word_weights_dict[w] for w in words]

    if clustering_type in clustering_dict.keys():
        # cluster words to cluster labels
        labels = clustering_dict[clustering_type](word_embeddings, word_weights, params)

        # assign each word to cluster list
        cluster_words = [[] for _ in range(len(set(labels)))]
        cluster_embeddings = [[] for _ in range(len(cluster_words))]
        for l_id, label in enumerate(list(labels)):

            w = words[l_id]
            if w not in vocab:
                continue

            cluster_words[label].append(w)
            cluster_embeddings[label].append(word_embeddings[l_id])

    else:
        assert clustering_type == "nmf"
        cluster_words, cluster_embeddings = nmf_clustering(word_embeddings, words, **params)

    # remove clusters with <= 5 words:
    cleaned_cluster_words = []
    cleaned_cluster_embeddings = []
    for i_c, c in enumerate(cluster_words):

        if len(c) <= 5:
            continue
        cleaned_cluster_words.append(c)
        cleaned_cluster_embeddings.append(cluster_embeddings[i_c])

    # if no clusters have >= 6 words
    if len(cleaned_cluster_words) == 0:
        cleaned_cluster_words.append([w for c in cluster_words for w in c])
        cleaned_cluster_embeddings.append([emb for c in cluster_embeddings for emb in c])

    if ranking_weight_type is None:
        return cleaned_cluster_words, cleaned_cluster_embeddings
    else:
        return sort_words(processed_docs, cleaned_cluster_words, cleaned_cluster_embeddings,
                          weight_type=ranking_weight_type)


def document_clustering(doc_data: list, doc_embeddings: list, vocab: list,
                        clustering_type: str, params: dict,
                        clustering_weight_type: str = 'vocab_count') -> (list, list):
    """
    word_clusters returns a sorted list of words for each cluster

    :param doc_data: list of preprocessed documents
    :param doc_embeddings: list of word embeddings
    :param vocab: list of vocabulary words
    :param clustering_type: defines the clustering method ('kmeans', 'agglomerative', 'spectral')
    :param params: clustering parameters
    :param clustering_weight_type: word weighting type used for clustering ("tf", "tf-df", "tf-idf")

    :return:
    """

    doc_names = [str(i) for i in range(len(doc_data))]

    assert len(doc_embeddings) == len(doc_names), "doc_embeddings and word list do not have the same length"
    assert clustering_type in ['kmeans', 'agglomerative'], "incorrect clustering_type"

    clustering_dict = {
        'kmeans': kmeans_clustering,
        'agglomerative': agglomerative_clustering,
        'spectral': spectral_clustering
    }

    # print("Performing weighted clustering!")
    # doc_weights_dict = get_doc_weights(doc_data, vocab, weight_type=clustering_weight_type)
    # doc_weights = [doc_weights_dict[doc] for doc in doc_names]
    doc_weights = None

    # cluster words to cluster labels
    labels = clustering_dict[clustering_type](doc_embeddings, doc_weights, params)

    # assign each word to cluster list
    cluster_docs = [[] for _ in range(len(set(labels)))]
    cluster_embeddings = [[] for _ in range(len(cluster_docs))]

    for l_id, label in enumerate(list(labels)):

        cluster_docs[label].append(doc_data[l_id])
        cluster_embeddings[label].append(doc_embeddings[l_id])

    # todo: implement NMF for document clustering

    assert len(labels) == len(doc_data)
    return cluster_docs, cluster_embeddings, labels, doc_data


if __name__ == "__main__":
    docs = [["a", "asa", "asa"], ["a", "a", "aa"]]

    # calculate words frequencies per document
    # word_frequencies_per_doc = [Counter(doc) for doc in docs]

