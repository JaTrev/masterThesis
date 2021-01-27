from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import math


def absolute_word_counter(processed_docs: list) -> Counter:
    temp_doc = []
    for doc in processed_docs:
        temp_doc.extend(doc)
    return Counter(temp_doc)


def get_doc_weights(processed_docs: list, vocab: list, weight_type: str = 'vocab_count') -> dict:

    doc_weight = {}

    if weight_type == "vocab_count":

        for i, doc in enumerate(processed_docs):
            doc_weight.update({str(i): len([w for w in doc if w in vocab])})

    assert len(doc_weight) == len(processed_docs)

    return doc_weight


def get_word_weights(processed_docs: list, vocab: list, n_words: int, weight_type: str = 'tf') -> dict:

    assert weight_type in ["tf", "tf-df", "tf-idf"], "weight_type not in ('tf', 'tf-df', 'tf-idf')"

    n_docs = len(processed_docs)
    absolute_counter = absolute_word_counter(processed_docs)

    word_weight = {}
    if weight_type == "tf":
        # calculate tf

        for w in vocab:
            word_weight.update({w: absolute_counter[w] / n_words})

    elif weight_type == "tf-df":
        # calculate tf-idf

        # calculate words frequencies per document
        word_frequencies_per_doc = [Counter(doc) for doc in processed_docs]

        # calculate document frequency
        words_per_doc = [list(word_frequency.keys()) for word_frequency in word_frequencies_per_doc]
        document_frequencies = Counter([w for doc in words_per_doc for w in doc])

        for w in vocab:
            df = document_frequencies[w] / n_docs
            tf = absolute_counter[w] / n_words
            word_weight.update({w: tf * df})

    elif weight_type == "tf-idf":
        # calculate tf-idf

        # calculate words frequencies per document
        word_frequencies_per_doc = [Counter(doc) for doc in processed_docs]

        # calculate document frequency
        words_per_doc = [list(word_frequency.keys()) for word_frequency in word_frequencies_per_doc]
        document_frequencies = Counter([w for doc in words_per_doc for w in doc])

        for w in vocab:
            idf = math.log(n_docs / (document_frequencies[w] + 1))
            tf = absolute_counter[w] / n_words
            word_weight.update({w: tf * idf})

    assert len(word_weight) == len(vocab)

    return word_weight


def get_most_similar_indices(embedding, list_embedding, n_most_similar: int = 10):

    sim_matrix = cosine_similarity(embedding.reshape(1, -1), list_embedding)[0]
    most_sim = np.argsort(sim_matrix, axis=None)[:: -1]

    return most_sim[:n_most_similar]
