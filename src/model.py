from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import LsiModel, LdaModel
from gensim import corpora
from gensim.models.wrappers import LdaMallet
from sklearn.decomposition import NMF


def lsa_topics(processed_data: list, n_topics: int = 10, n_words: int = 10):
    dictionary = corpora.Dictionary(processed_data)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_data]
    lsa_model = LsiModel(doc_term_matrix, num_topics=n_topics, id2word=dictionary)

    topics = []
    for topic_word_dist in lsa_model.get_topics():

        topic = [lsa_model.id2word[i] for i in topic_word_dist.argsort()[::-1][:n_words]]
        print(topic)
        topics.append(topic)

    return topics


def lda_topics(processed_data: list, n_topics: int = 10, learning_decay: float = 0.5,
               learning_offset: float = 1.0, max_iter: int = 50, n_words: int = 10):
    # TODO: add tf-idf

    dictionary = corpora.Dictionary(processed_data, )
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_data]

    lda_model = LdaModel(doc_term_matrix, id2word=dictionary, num_topics=n_topics, offset=learning_offset,
                         random_state=42, update_every=1, iterations=max_iter,
                         passes=10, alpha='auto', eta="auto", decay=learning_decay, per_word_topics=True)

    topics = []
    for i_t, topic_word_dist in enumerate(lda_model.get_topics()):
        topic = [lda_model.id2word[w_id] for w_id,_ in lda_model.get_topic_terms(i_t, topn=n_words)]
        print(topic)
        topics.append(topic)

    return topics


def lda_mallet_topics(processed_data: list, n_topics: int = 10, n_words: int = 10, n_iterations: int = 1000):
    mallet_path = 'data/mallet-2.0.8/bin/mallet'
    dictionary = corpora.Dictionary(processed_data, )
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_data]

    lda_mallet = LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=n_topics,
                           id2word=dictionary, iterations=n_iterations)

    topics = []
    for i_t, _ in enumerate(lda_mallet.get_topics()):
        topic = [w for w, _ in lda_mallet.show_topic(i_t, topn=n_words)]
        print(topic)
        topics.append(topic)

    return topics


def nmf_topics(preprocessed_data: list, n_topics: int = 10, n_features: int = 200,
               n_words:int = 10, init: str = 'nndsvd', solver: str = 'mu', beta_loss='kullback-leibler',use_tfidf: bool = True):

    assert init in [None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar'], "need an appropriate method to init to procedure"
    assert solver in ['mu', 'cd'], "need an appropriate solver"
    assert beta_loss in ['frobenius', 'kullback-leibler', 'itakura-saito'], "need an appropriate beta_loss"
    str_split = " "
    raw_docs = [str_split.join(doc) for doc in preprocessed_data]

    if use_tfidf:
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=n_features,
                                           stop_words='english')
        new_data = tfidf_vectorizer.fit_transform(raw_docs)
        feature_names = tfidf_vectorizer.get_feature_names()

    else:
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words='english')
        new_data = tf_vectorizer.fit_transform(raw_docs)
        feature_names = tf_vectorizer.get_feature_names()

    nmf_model = NMF(n_components=n_topics, init=init, beta_loss=beta_loss, solver=solver,
                    max_iter=1000, alpha=.1, l1_ratio=.5, random_state=42).fit(new_data)

    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        print(top_features)
        topics.append(top_features)
    return topics


