from src.get_data import *
from src.preprocessing import *
from src.model import *
from src.evaluation import *
from src.clustering import *
from src.vectorization import *
from src.visualizations import *
from src.bert import *


data, _ = get_data()

#TODO: create a main() function

if __name__ == "__main__":
    vocab = preprocessing(data, vocab=True)

    data_processed = preprocessing(data, do_lemmatizing=True)

    topics_nmf = nmf_topics(data_processed, n_topics=10, solver='cd', beta_loss='frobenius', use_tfidf=False)
    # topics_lsa = lsa_topics(data_processed, n_topics=10)
    # topics_lda = lda_topics(data_processed, n_topics=10)

    plt = create_circle_tree(topics_nmf)
    plt.show()
    assert 0

    topics_type = [topics_nmf, topics_lsa, topics_lda]

    topic_type_sizes = []
    for topic_tyle in topics_type:
        sizes = [len(t) for t in topics_type]
        topic_type_sizes.append(sizes)

    plt = box_plot(range(len(topic_type_sizes)), ys=topic_type_sizes, labels=["nmf", "lsa", "lda"],
                   x_label="Topic Types", y_label="Topic Sizes")
    plt.show()
    assert 0

    """
    x = list(range(2, 22, 2))
    y_nmf = []
    y_lsa = []
    y_lda = []
    y_lda_mallet = []
    for n_topic in x:
        topics = nmf_topics(data_processed, n_topics=n_topic, solver='cd', beta_loss='frobenius', use_tfidf=False)
        y_nmf.append(coherence_score(data_processed, topics))

        topics = lsa_topics(data_processed, n_topics=n_topic)
        y_lsa.append(coherence_score(data_processed, topics))
        
        topics = lda_topics(data_processed, n_topics=n_topic)
        y_lda.append(coherence_score(data_processed, topics))
        
        topics = lda_mallet_topics(data_processed, n_topics=n_topic)
        y_lda_mallet.append(coherence_score(data_processed, topics))
        
    """

    """
    topics = nmf_topics(data_processed, n_topics=10, solver='cd', beta_loss='frobenius', use_tfidf=False)
    print("----")
    print("nmf coherence score (tf, cd, frobenius):" + str(coherence_score(data_processed, topics)))
    print("----")

    print()
    topics = nmf_topics(data_processed, use_tfidf=True)
    print("----")
    print("nmf coherence score (tfidf):" + str(coherence_score(data_processed, topics)))
    print("----")

    print()
    topics = lsa_topics(data_processed, n_topics=10)
    print("----")
    print("lsa coherence score:" + str(coherence_score(data_processed, topics)))
    print("----")

    print()
    topics = lda_topics(data_processed, n_topics=10)
    print("----")
    print("lda coherence score:" + str(coherence_score(data_processed, topics)))
    print("----")

    print()
    topics = lda_mallet_topics(data_processed)
    print("----")
    print("lda mallet coherence score:" + str(coherence_score(data_processed, topics)))
    print("----")
    print("------------------------------")
    """


    ys = [y_nmf, y_lsa, y_lda, y_lda_mallet]
    plt = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score",
                       color_legends=["NMF", "LSA", "LDA", "LDA MALLET"])
    plt.show()

    # words, word_embeddings, w2v_model = get_word_vectors([], vocab, "data/w2v_node2vec")

    # word_similarities = get_word_similarities(word_embeddings)

    """
    clusters_words, clusters_embeddings = word_clusters(data_processed, words,
                                                        word_embeddings, clustering_type="kmeans",
                                                        params={'n_clusters': 10, 'random_state': 42, })
                                                        

    tsne_plot(clusters_words, clusters_embeddings)

    create_circle_tree(clusters_words)

    print(coherence_score(data_processed, clusters_words))
    """



