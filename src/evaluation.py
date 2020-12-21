from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import sklearn.metrics


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


def davies_bouldin_index(topic_words: list) -> float:
    """
    davies_bouldin_index calculates the davies_bouldin_score based on the topic words

    :param topic_words: list of words for each topic
    :return: davies_bouldin_index
    """

    temp_topic_words = []
    temp_labels = []

    for i_t, t_words in enumerate(topic_words):

        temp_labels.append(i_t)
        temp_topic_words.extend(t_words)

    return sklearn.metrics.davies_bouldin_score(temp_topic_words, temp_labels)
