from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def lda_topics(processed_data: list, tf_idf_flag: bool = False, n_topics: int = 10, learning_decay: float = 0.7,
               learning_offset: float = 10., max_iter: int = 50, top_n_word: int = 10):

    list_sep = " "
    data_string = [list_sep.join(l) for l in processed_data]

    if tf_idf_flag:
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = tfidf_vectorizer.fit_transform(data_string)
        feature_names = tfidf_vectorizer.get_feature_names()

    else:
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = tf_vectorizer.fit_transform(data_string)
        feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, learning_method='online',
                                    learning_offset=learning_offset, random_state=42,
                                    learning_decay=learning_decay).fit(doc_term_matrix)
    topics = []
    for topic_word_dist in lda.components_:
        topics.append([feature_names[i] for i in topic_word_dist.argsort()[::-1][:top_n_word]])

    return topics


