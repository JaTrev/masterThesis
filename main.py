from src.get_data import *
from src.preprocessing import *
from src.model import *
from src.evaluation import *
from src.clustering import *
from src.vectorization import *


data, _ = get_data()

#TODO: create a main() function

if __name__ == "__main__":
    data_processed = preprocessing(data)
    vocab = preprocessing(data, True)

    words, word_embeddings, w2v_model = get_word_vectors([], vocab, "data/w2v_node2vec")

    word_similarities = get_word_similarities(word_embeddings)

    clusters = word_clusters(data_processed, words, word_similarities, clustering_type="kmeans",
                             params={'n_clusters': 10, 'random_state': 42, })

    print(coherence_score(data_processed, clusters))



