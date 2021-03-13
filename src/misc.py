from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import math
from src.visualizations import *


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



def save_model_scores(models: list, model_topics: dict, model_c_v_scores: dict, model_npmi_scores: dict,
                      model_c_v_test_scores: dict, model_npmi_test_scores: dict, filename_prefix: str,
                      model_dbs_scores: dict = None):

    """

    :param models:
    :param model_topics:
    :param model_c_v_scores:
    :param model_npmi_scores:
    :param model_c_v_test_scores:
    :param model_npmi_test_scores:
    :param filename_prefix:
    :param model_dbs_scores:
    :return:
    """

    # if the same models saved in every score dict
    assert all(x in models for x in model_c_v_scores.keys())
    assert all(x in models for x in model_npmi_scores.keys())
    assert all(x in models for x in model_npmi_test_scores.keys())
    assert all(x in models for x in model_c_v_test_scores.keys())
    assert all(x in models for x in model_topics.keys())

    if isinstance(model_dbs_scores, dict):
        assert all(x in models for x in model_dbs_scores.keys())

    # c_v coherence score figure - intrinsic
    ys = [l for l in model_c_v_scores.values()]
    _, fig = scatter_plot(models, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=models, type='c_v')
    fig.savefig("visuals/" + filename_prefix + "_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)
    fig.close()

    # NPMI coherence score figure - intrinsic
    ys = [l for l in model_npmi_scores.values()]
    _, fig = scatter_plot(models, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=models, type='c_npmi')
    fig.savefig("visuals/" + filename_prefix + "_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # c_v coherence score figure - extrinsic
    ys = [l for l in model_c_v_test_scores.values()]
    _, fig = scatter_plot(models, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=models, type='c_v')
    fig.savefig("visuals/" + filename_prefix + "_extrinsic_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # NPMI coherence score figure - extrinsic
    ys = [l for l in model_npmi_test_scores.values()]
    _, fig = scatter_plot(models, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=models, type='c_npmi')
    fig.savefig("visuals/" + filename_prefix + "_extrinsic_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # save all topics with their associated scores
    for m in models:
        vis_topics_score(model_topics[m], model_c_v_scores[m], model_npmi_scores[m], model_c_v_test_scores[m],
                         model_npmi_test_scores[m], "visuals/ws_clusters_eval_" + str(m) + ".txt",
                         dbs_scores=model_dbs_scores[m])

    plt.close('all')
