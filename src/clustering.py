from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from collections import Counter


def tf_word_counter(processed_docs: list) -> Counter:
    temp_doc = []
    for doc in processed_docs:
        temp_doc.extend(doc)
    return Counter(temp_doc)


def kmeans_clustering(word_embeddings: list, word_weights: list = None, params: dict = None) -> list:
    model = KMeans(**params)
    return model.fit_predict(word_embeddings, word_weights)


def agglomerative_clustering(word_embeddings: list, word_weights: list = None, params: dict = None):
    model = AgglomerativeClustering(**params)
    return model.fit_predict(word_embeddings, word_weights)


def spectral_clustering(word_embeddings: list, word_weights :list = None, params: dict = None):
    model = SpectralClustering(**params)
    return model.fit_predict(word_embeddings, word_weights)


def word_clusters(processed_docs: list, words: list, word_embeddings: list,
                  clustering_type: str, params: dict, word_weights: list = None, n_words : int = 10) -> (list, list):
    """
    word_clusters returns a sorted list of words for each cluster

    :param processed_docs: list of preprocessed documents
    :param words: list of words
    :param word_embeddings: list of word embeddings
    :param clustering_type: defines the clustering method ('kmeans', 'agglomerative', 'spectral')
    :param params: clustering parameters
    :param word_weights: word weighting used for clustering
    :param n_words: number of words for every cluster
    :return:
    """

    assert len(word_embeddings) == len(words), "word_embeddings and word list do not have the same length"

    clustering_dict = {
        'kmeans': kmeans_clustering,
        'agglomerative': agglomerative_clustering,
        'spectral': spectral_clustering
    }
    assert clustering_type in ['kmeans', 'agglomerative', 'spectral'], "incorrect clustering_type"

    # cluster words to cluster labels
    labels = clustering_dict[clustering_type](word_embeddings, word_weights, params)

    # assign each word to cluster list
    cluster_words = [[] for _ in range(len(set(labels)))]
    cluster_embeddings = [[] for _ in range(len(cluster_words))]
    for l_id, l in enumerate(list(labels)):
        cluster_words[l].append(words[l_id])
        cluster_embeddings[l].append(word_embeddings[l_id])

    # sort cluster_words
    word_counter = tf_word_counter(processed_docs)
    sorted_cluster_words = []
    sorted_cluster_embeddings = []
    sorted_cluster_idxs = [sorted(range(len(c)), key=lambda k: word_counter[c[k]], reverse=True)[:n_words] for c
                           in cluster_words]

    for i_c in range(len(cluster_words)):
        sorted_cluster_words.append([cluster_words[i_c][i] for i in sorted_cluster_idxs[i_c]])
        sorted_cluster_embeddings.append([cluster_embeddings[i_c][i] for i in sorted_cluster_idxs[i_c]])

    return sorted_cluster_words, sorted_cluster_embeddings
