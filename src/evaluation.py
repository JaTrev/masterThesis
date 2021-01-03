from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import sklearn.metrics
from scipy.spatial import distance
import numpy as np


def coherence_score(processed_data: list, topic_words: list, cs_type: str = 'c_v', top_n_words: int = 10) -> float:
    """
    coherence_score calculates the coherence score based on the cluster_words and top_n_words.

    :param processed_data: list of processed documents
    :param topic_words:  list of words for each topic (sorted)
    :param cs_type: type of coherence score ('c_v' or 'u_mass')
    :param top_n_words: max. number of words used in each list of topic words
    :return: coherence score
    """

    assert cs_type in ['c_v', 'u_mass'], "the cs_type must either be 'c_v' or 'u_mass'"
    assert len(topic_words) > 1, "must be more than 1 topic"

    dictionary = corpora.Dictionary(processed_data)
    corpus = [dictionary.doc2bow(text) for text in processed_data]

    cm = CoherenceModel(topics=topic_words,
                        corpus=corpus,
                        dictionary=dictionary,
                        texts=processed_data,
                        coherence=cs_type,
                        topn=top_n_words)

    return cm.get_coherence()


def davies_bouldin_index(topic_word_embeddings: list) -> float:
    """
    davies_bouldin_index calculates the davies_bouldin_score based on the topic word embeddings

    :param topic_word_embeddings: list of words for each topic
    :return: davies_bouldin_index
    """

    temp_topic_words_embeddings = []
    temp_labels = []

    for i_t, t_word_embeddings in enumerate(topic_word_embeddings):

        temp_labels.extend([i_t] * len(t_word_embeddings))
        temp_topic_words_embeddings.extend(t_word_embeddings)

    return sklearn.metrics.davies_bouldin_score(temp_topic_words_embeddings, temp_labels)


def compute_aic_bic(data_matrix, labels) -> (float, float):
    """
    compute_aic_bic calculates AIC and BIC, the the data and the labels

    Using pseudo algorithm:
    https://stats.stackexchange.com/questions/55147/compute-bic-clustering-criterion-to-validate-
    clusters-after-k-means/55160#55160

    Used here:
    https://stats.stackexchange.com/questions/374002/best-bic-value-for-k-means-clusters/374013#374013
    https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans

    general discussion about aic, bic
    https://stats.stackexchange.com/questions/577/is-there-any-reason-to-prefer-the-aic-or-bic-over-the-other

    :param data_matrix: embeddings (N x P)
    :param labels: label for each data point N x 1
    :return: aic, bic
    """
    # number of samples
    n = len(data_matrix)

    # dimension size
    p = len(data_matrix[0])

    # number of clusters
    k = len(set(labels))

    # 1. Compute 1 x K row n_c showing number of objects in each cluster.
    n_c = np.bincount(labels)

    # 2. Compute P x K matrix v_c containing variances by clusters.
    #    Use denominator "n", not "n-1", to compute those, because there may be clusters with just one object.
    v_c = []
    # variance for each dimension i_p
    for i_p in range(len(p)):

        dimension_var = []
        # variance for cluster i_k
        for i_k in range(len(k)):

            temp_var = np.var(data_matrix[np.where(labels == i_k)][i_p], ddof=0)
            dimension_var.append(temp_var)

        v_c.append(dimension_var)
    assert len(v_c) == p
    assert len(v_c[0]) == k

    # 3. Compute P x 1 column containing variances for the whole sample. Use "n-1" denominator.
    #    Then propagate the column to get P x K matrix V.
    p_v = []
    for i_p in range(len(p)):
        p_v.append(np.var(data_matrix[:, i_p], ddof=0))
    assert len(p_v) == p

    v = []
    for i_p in range(len(p)):
        v.append([p_v[i_p]] * k)
    assert len(v) == p
    assert len(v[0]) == k

    # 4. Compute log-likelihood LL, 1 x K row. LL = -n_c &* csum( ln(Vc + V)/2 ),
    #    where "&*" means usual, elementwise multiplication;
    #    "csum" means sum of elements within columns.
    temp_matrix = np.ln(v_c + v)/2
    ll = np.multiply(-n_c, [sum(temp_matrix[:, i_c]) for i_c in temp_matrix[0]])

    bic = -2 * sum(ll) + 2 * k * p * np.ln(n)

    aic = -2 * sum(ll) + 4*k*p

    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)
    """

    return aic, bic
