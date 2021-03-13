from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import sklearn.metrics
from sklearn.metrics import adjusted_rand_score, accuracy_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score
import numpy as np


def c_v_coherence_score(processed_data: list, topic_words: list, cs_type: str = 'c_v', top_n_words: int = 10) -> float:
    """
    c_v_coherence_score calculates the coherence score based on the cluster_words and top_n_words.

    :param processed_data: list of processed documents
    :param topic_words:  list of words for each topic (sorted)
    :param cs_type: type of coherence score ('c_v' or 'c_npmi')
    :param top_n_words: max. number of words used in each list of topic words
    :return: coherence score
    """

    assert len(topic_words) > 1, "need at least 2 topics"

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

    return float("{:.2f}".format(cm.get_coherence()))


def npmi_coherence_score(documents: list, topic_words_large: list, ntopics: int):
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

    return float("{:.2f}".format(avg_score))


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

    return float("{:.2f}".format(sklearn.metrics.davies_bouldin_score(temp_topic_words_embeddings, temp_labels)))


def ari_score(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)


def acc_score(labels_true, labels_pred, normalize=True, sample_weight=None):
    return accuracy_score(labels_true, labels_pred, normalize, sample_weight)


def ami_score(labels_true, labels_pred, average_method='arithmetic'):
    return adjusted_mutual_info_score(labels_true, labels_pred, average_method=average_method)


def nmi_score(labels_true, labels_pred, average_method='arithmetic'):
    return normalized_mutual_info_score(labels_true, labels_pred, average_method=average_method)