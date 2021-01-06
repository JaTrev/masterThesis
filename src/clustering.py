from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from src.misc import *
import numpy as np


def kmeans_clustering(word_embeddings: list, word_weights: list = None, params: dict = None) -> list:
    model = KMeans(**params)
    return model.fit_predict(word_embeddings, word_weights)


def agglomerative_clustering(word_embeddings: list, word_weights: list = None, params: dict = None):
    model = AgglomerativeClustering(**params)
    return model.fit_predict(word_embeddings, word_weights)


def spectral_clustering(word_embeddings: list, word_weights: list = None, params: dict = None):
    model = SpectralClustering(**params)
    return model.fit_predict(word_embeddings, word_weights)


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
                  ranking_weight_type: str = 'tf') -> (list, list):
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
    assert clustering_type in ['kmeans', 'agglomerative', 'spectral'], "incorrect clustering_type"
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

    # cluster words to cluster labels
    labels = clustering_dict[clustering_type](word_embeddings, word_weights, params)

    # assign each word to cluster list
    cluster_words = [[] for _ in range(len(set(labels)))]
    cluster_embeddings = [[] for _ in range(len(cluster_words))]
    for l_id, l in enumerate(list(labels)):

        w = words[l_id]
        if w not in vocab:
            continue

        cluster_words[l].append(w)
        cluster_embeddings[l].append(word_embeddings[l_id])

    # remove clusters with < 5 words:
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

    return sort_words(processed_docs, cleaned_cluster_words,
                      cleaned_cluster_embeddings, weight_type=ranking_weight_type)


if __name__ == "__main__":
    docs = [["a", "asa", "asa"], ["a", "a", "aa"]]

    # calculate words frequencies per document
    # word_frequencies_per_doc = [Counter(doc) for doc in docs]

    # calculate document frequency
