from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import sklearn.metrics
from sklearn.metrics import adjusted_rand_score, accuracy_score, adjusted_mutual_info_score
from scipy.spatial import distance
from collections import Counter
import numpy as np


def coherence_score(processed_data: list, topic_words: list, cs_type: str = 'c_v', top_n_words: int = 10) -> float:
    """
    coherence_score calculates the coherence score based on the cluster_words and top_n_words.

    :param processed_data: list of processed documents
    :param topic_words:  list of words for each topic (sorted)
    :param cs_type: type of coherence score ('c_v' or 'c_npmi')
    :param top_n_words: max. number of words used in each list of topic words
    :return: coherence score
    """

    assert cs_type in ['c_v', 'c_npmi'], "the cs_type must either be 'c_v' or 'c_npmi'"
    assert len(topic_words) >= 1, "need at least 1 topic"

    if len(topic_words) == 1:
        return -1000

    dictionary = corpora.Dictionary(processed_data)
    corpus = [dictionary.doc2bow(text) for text in processed_data]

    dictionary_words = dictionary.token2id
    new_topics = []
    for topic in topic_words:

        temp_topic = []
        for w in topic:
            if w in dictionary_words:
                temp_topic.append(w)

            if len(temp_topic) == 10:
                break
        new_topics.append(temp_topic)

    cm = CoherenceModel(topics=new_topics,
                        # corpus=corpus,
                        dictionary=dictionary,
                        texts=processed_data,
                        coherence=cs_type,
                        topn=top_n_words)

    return cm.get_coherence()


def average_npmi_topics(documents, topic_words_large, ntopics,):
    """
    Average NPMI from:
    Tired of Topic Models? Clusters of Pretrained Word Embeddings Make for Fast and Good Topics too!
    by Suzanna Sia et al
    Github: https://github.com/adalmia96/Cluster-Analysis
    :param topic_words:
    :param ntopics:
    :param word_doc_counts:
    :param nfiles:
    :return:
    """

    if ntopics == 1:
        return -1000

    eps = 10**(-12)
    n_docs = len(documents)
    topic_words = [t[:10] for t in topic_words_large]

    word_to_doc = {}
    all_cluster_words = [w for t in topic_words for w in t]
    for i_d, doc in enumerate(documents):

        for w in all_cluster_words:

            if w in doc:

                if w in word_to_doc:
                    word_to_doc[w].add(i_d)

                else:
                    word_to_doc[w] = set()
                    word_to_doc[w].add(i_d)

    all_topics = []
    for k in range(ntopics):
        word_pair_counts = 0
        topic_score = []

        ntopw = len(topic_words[k])

        for i in range(ntopw-1):
            for j in range(i+1, ntopw):
                w1 = topic_words[k][i]
                w2 = topic_words[k][j]

                w1_dc = len(word_to_doc.get(w1, set()))
                # len(word_doc_counts.get(w1, set()))

                w2_dc = len(word_to_doc.get(w2, set()))
                # len(word_doc_counts.get(w2, set()))

                w1w2_dc = len(word_to_doc.get(w1, set()) & word_to_doc.get(w2, set()))
                # len(word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set()))

                # what we had previously:
                #pmi_w1w2 = np.log(((w1w2_dc * nfiles) + eps) / ((w1_dc * w2_dc) + eps))

                # Correct eps:
                pmi_w1w2 = np.log((w1w2_dc * n_docs) / ((w1_dc * w2_dc) + eps) + eps)
                npmi_w1w2 = pmi_w1w2 / (- np.log( (w1w2_dc)/n_docs + eps))

                # Sanity check Which is equivalent to this:
                #if w1w2_dc ==0:
                #    npmi_w1w2 = -1
                #else:
                    #pmi_w1w2 = np.log( (w1w2_dc * nfiles)/ (w1_dc*w2_dc))
                    #npmi_w1w2 = pmi_w1w2 / (-np.log(w1w2_dc/nfiles))

                #if npmi_w1w2>1 or npmi_w1w2<-1:
                #    print("NPMI score not bounded for:", w1, w2)
                #    print(npmi_w1w2)
                #    sys.exit(1)

                topic_score.append(npmi_w1w2)

        all_topics.append(np.mean(topic_score))

    avg_score = np.around(np.mean(all_topics), 5)

    return avg_score


def davies_bouldin_index(topic_word_embeddings: list) -> float:
    """
    davies_bouldin_index calculates the davies_bouldin_score based on the topic word embeddings

    :param topic_word_embeddings: list of words for each topic
    :return: davies_bouldin_index
    """

    if len(topic_word_embeddings) == 1:
        return -1000

    temp_topic_words_embeddings = []
    temp_labels = []

    for i_t, t_word_embeddings in enumerate(topic_word_embeddings):

        temp_labels.extend([i_t] * len(t_word_embeddings))
        temp_topic_words_embeddings.extend(t_word_embeddings)

    return sklearn.metrics.davies_bouldin_score(temp_topic_words_embeddings, temp_labels)


def ari_score(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)


def acc_score(labels_true, labels_pred, normalize=True, sample_weight=None):
    return accuracy_score(labels_true, labels_pred, normalize, sample_weight)


def ami_score(labels_true, labels_pred, average_method='arithmetic'):
    return adjusted_mutual_info_score(labels_true, labels_pred, average_method)


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
    labels  = kmeans.labels
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
