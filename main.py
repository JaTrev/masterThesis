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
    vocab = preprocessing(data, True)

    data_processed = preprocessing(data)

    words, word_embeddings, w2v_model = get_word_vectors([], vocab, "data/w2v_node2vec")

    # word_similarities = get_word_similarities(word_embeddings)

    clusters_words, clusters_embeddings = word_clusters(data_processed, words,
                                                        word_embeddings, clustering_type="kmeans",
                                                        params={'n_clusters': 10, 'random_state': 42, })

    tsne_plot(clusters_words, clusters_embeddings)

    create_circle_tree(clusters_words)

    print(coherence_score(data_processed, clusters_words))



