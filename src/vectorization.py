from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def get_saved_w2v_model(w2v_model: str) -> Word2Vec:
    """
    get_saved_w2v_model returns a pretrained word2vec model

    :param w2v_model: name of the pretrained model

    :return:  word2vec model
    """
    w2v_model = Word2Vec.load(w2v_model)
    return w2v_model


def create_w2v_model(processed_data: list, min_c: int, win: int, negative: int, ns_exponent: float,
                     sample: float = 6e-5, alpha: float = 0.03, min_alpha: float = 0.0007,
                     epochs: int = 30, size: int = 300):

    w2v_model = Word2Vec(min_count=min_c,
                         window=win,
                         size=size,
                         sample=sample,
                         alpha=alpha,
                         min_alpha=min_alpha,
                         negative=negative,
                         ns_exponent=ns_exponent,
                         sorted_vocab=1,
                         compute_loss=True)
    w2v_model.build_vocab(processed_data, progress_per=10000)
    w2v_model.train(processed_data,
                    total_examples=w2v_model.corpus_count,
                    epochs=epochs, report_delay=1)

    # normalize vectors:
    w2v_model.init_sims(replace=True)

    return w2v_model


def get_word_vectors(processed_data: list, vocab: list, saved_model=None, params=None) -> (list, list, Word2Vec):
    """
    get_word_vectors calculates the vector representations of words

    :param processed_data: list of processed documents
    :param processed_data: list of words in the processed documents
    :param saved_model: name of a previously saved w2v_model
    :param params: parameters for a w2v_model

    :return: list of words, list of word embeddings, w2v_model
    """

    if isinstance(saved_model, str):
        w2v_model = get_saved_w2v_model(saved_model)

    else:

        assert isinstance(params, dict), "missing w2v_model params"

        assert {'processed_data', 'min_c', 'negative', 'ns_exponent'}.issubset(params.keys()), (
            "missing w2v_model params, need: 'processed_data', 'min_c', 'negative', 'ns_exponent'")

        w2v_model = create_w2v_model(processed_data, **params)

    vocab_words = [w for w in w2v_model.wv.index2word if w in vocab]
    vocab_embeddings = [w2v_model.wv.vectors[i_w] for i_w, w in enumerate(w2v_model.wv.index2word)
                        if w in vocab]

    return vocab_words, vocab_embeddings, w2v_model


def get_word_similarities(word_embeddings: list) -> list:
    similarities = cosine_similarity(word_embeddings)

    assert len(similarities) == len(word_embeddings)

    return similarities


def get_tf_idf(processed_data: list):
    str_separator = " "
    data_strings = [str_separator.join(doc) for doc in processed_data]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_strings)

    return tfidf_matrix


if __name__ == "__main__":
    # w2v_model = Word2Vec.load("w2v_node2vec")
    word, word_embeddings, w2v_model_test = get_word_vectors([], "data/w2v_node2vec")
