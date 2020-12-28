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

    print("------------------------------")
    topics = nmf_topics(data_processed, solver='cd', beta_loss='frobenius', use_tfidf=False)
    print("----")
    print("nmf coherence score (tf, cd, frobenius):" + str(coherence_score(data_processed, topics)))
    print("----")

    print()
    topics = nmf_topics(data_processed, solver='mu', use_tfidf=False)
    print("----")
    print("nmf coherence score (tf, mu):" + str(coherence_score(data_processed, topics)))
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



