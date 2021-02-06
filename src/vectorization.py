from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import numpy as np
import os
import pickle
import io


def get_saved_w2v_model(w2v_model: str) -> Word2Vec:
    """
    get_saved_w2v_model returns a pretrained word2vec model

    :param w2v_model: name of the pretrained model

    :return:  word2vec model
    """
    w2v_model = Word2Vec.load(w2v_model)
    return w2v_model


def create_w2v_model(processed_data: list, min_c: int, win: int, negative: int, seed: int,
                     sample: float = 6e-5, alpha: float = 0.03, min_alpha: float = 0.0007, ns_exponent: float = 0.75,
                     epochs: int = 30, size: int = 300, sg=0, cbow_mean=1, hs=0):

    w2v_model = Word2Vec(min_count=min_c,
                         window=win,
                         size=size,
                         sample=sample,
                         alpha=alpha,
                         min_alpha=min_alpha,
                         negative=negative,
                         ns_exponent=ns_exponent,
                         sorted_vocab=1,
                         seed=seed,
                         compute_loss=True,
                         workers=1,
                         sg=sg,
                         cbow_mean=cbow_mean,
                         hs=hs
                         )
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
    :param vocab: list of words in the processed documents
    :param saved_model: name of a previously saved w2v_model
    :param params: parameters for a w2v_model

    :rtype: list, list, Word2Vec
    :return: vocab_words, vocab_embeddings, w2v_model
    """

    if isinstance(saved_model, str):
        w2v_model = get_saved_w2v_model(saved_model)

    else:

        assert isinstance(params, dict), "missing w2v_model params"

        assert {'min_c', 'win', 'negative', 'seed'}.issubset(params.keys()), (
            "missing w2v_model params, need: min_c', 'win', 'negative',', 'seed'")

        w2v_model = create_w2v_model(processed_data, **params)

    # vocab_words and vocab_embeddings are sorted like vocab
    vocab_words = [w for w in vocab if w in w2v_model.wv.index2word]
    vocab_embeddings = [w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)]
                        for w in vocab_words]

    return vocab_words, vocab_embeddings, w2v_model


def get_avg_sentence_doc_embeddings_w2v(original_data: list, nodes: list, vocab: list, vocab_embeddings):
    # calculate sentence embeddings based on vocab avg

    data_sentences = []
    sentence_embeddings = []

    data_docs = []
    doc_embeddings = []

    for doc in original_data:
        sents = doc.split(" . ")

        doc_embedds_list = []

        for sent in sents:
            sent_words = sent.lower().split()

            sent_embedds_list = [vocab_embeddings[vocab.index(w)] for w in sent_words if w in vocab]

            # sentence has at least 1 word
            if len(sent_embedds_list) > 1:
                sent_embedding = np.average(sent_embedds_list, axis=0)
            elif len(sent_embedds_list) == 1:
                sent_embedding = sent_embedds_list[0]

            sentence_embeddings.append(sent_embedding)
            data_sentences.append(sent_words)

            doc_embedds_list.append(sent_embedding)

        # doc has to have at least 1 sentence
        if len(doc_embedds_list) > 0:

            data_docs.append(doc.lower())

            if len(doc_embedds_list) == 1:
                doc_embeddings.append(doc_embedds_list[0])

            elif len(doc_embedds_list) > 1:
                doc_embeddings.append(np.average(doc_embedds_list, axis=0))

    node_sentence_embeddings = {}
    node_doc_embeddings = {}
    for node in nodes:

        # calculate sentence embeddings for each word (node)
        sent_ids = [i_s for i_s, s in enumerate(data_sentences) if node in s]

        sents_embds = [sentence_embeddings[sent_id] for sent_id in sent_ids]

        if len(sents_embds) == 1:
            node_sentence_embeddings[node] = sents_embds[0]

        elif len(sents_embds) > 1:
            node_sentence_embeddings[node] = np.average(sents_embds, axis=0)
        else:
            print("missing node's sentence embedding (in get_avg_sentence_doc_embeddings_w2v): " + str(node))
            assert 0

        # calculate doc embeddings for each word (node)
        doc_ids = [i_d for i_d, doc in enumerate(data_docs) for sent in doc.split() if node in sent]
        doc_embds = [doc_embeddings[doc_id] for doc_id in doc_ids]

        if len(doc_embds) == 1:
            node_doc_embeddings[node] = doc_embds[0]
        elif len(doc_embds) > 1:
            node_doc_embeddings[node] = np.average(doc_embds, axis=0)
        else:
            print("missing node's doc embedding (in get_avg_sentence_doc_embeddings_w2v): " + str(node))
            assert 0

    assert len(node_doc_embeddings) == len(nodes)
    assert len(node_sentence_embeddings) == len(nodes)

    return node_sentence_embeddings, node_doc_embeddings


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


def get_glove_embeddings(vocab: list, filename=None, save_embeddings=False):

    if isinstance(filename, str):

        with open(filename, "rb") as f:
            embeddings_index = pickle.load(f)

    else:
        # get glove embeddings
        print("Getting GloVe embeddings!")

        embeddings_index = {}
        f = open(os.path.join('data/', 'glove.twitter.27B.200d.txt'), encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]

            try:
                embeddings_index[word] = np.asarray(values[1:], dtype='float32')
            except ValueError:
                print("couldn't include:")
                print(word)
        f.close()

    glove_words = list(embeddings_index.keys())
    vocab_embeddings = []
    new_vocab = []

    for w in vocab:
        if w in glove_words:

            new_vocab.append(w)
            vocab_embeddings.append(embeddings_index[w])

    print("glove vocab has " + str(len(vocab) - len(new_vocab)) + " words less")

    if save_embeddings:
        assert isinstance(filename, str), "have not specified where to save the embeddings"

        with open(filename, "wb") as myFile:
            pickle.dump(embeddings_index, myFile)

    return new_vocab, vocab_embeddings


def get_fast_text_embeddings(processed_data: list, vocab: list, min_c: int = 50, win: int = 5, negative: int = 60,
                             sample: float = 6e-5, alpha: float = 0.03,
                             min_alpha: float = 0.0007, epochs: int = 30, size: int = 300):

    # model = FastText(size=300, negative=40)
    model = FastText(min_count=min_c, window=win, size=size, sample=sample, alpha=alpha, min_alpha=min_alpha,
                     negative=negative, sorted_vocab=1)

    model.build_vocab(processed_data, progress_per=10000)
    model.train(processed_data, total_examples=model.corpus_count, epochs=epochs, report_delay=1)

    # normalize vectors:
    model.init_sims(replace=True)

    vocab_words = [w for w in vocab if w in model.wv.index2word]
    vocab_embeddings = [model.wv.vectors[model.wv.index2word.index(w)]
                        for w in vocab if w in model.wv.index2word]

    return vocab_words, vocab_embeddings


def get_pretrained_fast_text_embeddings(vocab: list):
    print("Getting fastText Embeddings")
    file_name = "data/wiki-news-300d-1M.vec"
    # "data/crawl-300d-2M.vec"

    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    fast_test_embeddings = {}

    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        fast_test_embeddings[tokens[0]] = map(float, tokens[1:])

    print("Got Embeddings")

    glove_words = list(fast_test_embeddings.keys())

    vocab_embeddings = []
    new_vocab = []

    for w in tqdm(vocab):
        if w in glove_words:
            new_vocab.append(w)
            vocab_embeddings.append(fast_test_embeddings[w])

    print("fastText vocab has " + str(len(vocab) - len(new_vocab)) + " words less")

    return new_vocab, vocab_embeddings


def get_doc_embeddings(processed_data: list, data_labels: list, vocab: list, embedding_type: str, params=None, saved_model=None):
    assert len(processed_data) == len(data_labels)

    doc_embeddings = []

    if embedding_type == "w2v_avg":
        vocab_words, vocab_embeddings, _ = get_word_vectors(processed_data=processed_data, vocab=vocab,
                                                            saved_model=saved_model,  params=params)

        doc_data = []
        doc_labels = []
        for i, doc in enumerate(processed_data):
            if any([w in doc for w in vocab_words]):
                doc_data.append(doc)
                doc_labels.append(data_labels[i])

        for i, doc in enumerate(doc_data):
            temp_embeddings = [vocab_embeddings[vocab_words.index(w)] for w in doc if w in vocab_words]

            if len(temp_embeddings) > 1:
                doc_embeddings.append(np.mean(temp_embeddings, axis=0))

            elif len(temp_embeddings) == 1:
                doc_embeddings.append(temp_embeddings[0])
            else:
                print("error")
                print([w for w in doc if w in vocab_words])
                print("---------")
                continue

    elif embedding_type == "w2v_sum":
        vocab_words, vocab_embeddings, _ = get_word_vectors(processed_data=processed_data, vocab=vocab, params=params)

        doc_data = []
        doc_labels = []
        for i, doc in enumerate(processed_data):
            if any([w in doc for w in vocab_words]):
                doc_data.append(doc)
                doc_labels.append(data_labels[i])

        for i, doc in enumerate(doc_data):
            temp_embeddings = [vocab_embeddings[vocab_words.index(w)] for w in doc if w in vocab_words]

            if len(temp_embeddings) > 1:
                doc_embeddings.append(np.sum(temp_embeddings, axis=0))

            elif len(temp_embeddings) == 1:
                doc_embeddings.append(temp_embeddings[0])
            else:
                print("error")
                print([w for w in doc if w in vocab_words])
                print("---------")
                continue

    else:
        assert embedding_type == "doc2vec"

        doc_data = []
        doc_labels = []
        for i, doc in enumerate(processed_data):
            if any([w in doc for w in vocab]):
                doc_data.append(doc)
                doc_labels.append(data_labels[i])
        vocab_words, vocab_embeddings, doc_embeddings, _ = get_doc2vec_embeddings(doc_data, vocab, **params)

    assert all([len(doc_embeddings[0]) == len(e) for e in doc_embeddings])
    assert len(doc_data) == len(doc_embeddings)
    assert len(doc_data) == len(doc_labels)
    return doc_data, doc_labels, doc_embeddings, vocab_words, vocab_embeddings


def get_avg_sentence_doc_embeddings_w2v_2(original_data: list, nodes: list, vocab: list, vocab_embeddings):
    # calculate sentence embeddings based on vocab avg
    # nodes have indices not words

    data_sentences = []
    sentence_embeddings = []

    data_docs = []
    doc_embeddings = []

    for doc in original_data:
        sents = doc.split(" . ")

        temp_doc_embedds = []

        for sent in sents:
            sent_lower = sent.lower().split()

            temp = [vocab_embeddings[vocab.index(w)] for w in sent_lower if w in vocab]

            # sentence has to have more than 1 word
            if len(temp) > 1:
                sent_embedding = np.average(temp, axis=0)

                sentence_embeddings.append(sent_embedding)
                data_sentences.append(sent_lower)

                temp_doc_embedds.append(sent_embedding)

        # doc has to have at least 1 sentence
        if len(temp_doc_embedds) > 0:
            data_docs.append(doc.lower())

            if len(temp_doc_embedds) == 1:
                doc_embeddings.append(temp_doc_embedds[0])

            elif len(temp_doc_embedds) > 1:
                doc_embeddings.append(np.average(temp_doc_embedds, axis=0))

    node_sentence_embeddings = {}
    node_doc_embeddings = {}
    for node in nodes:

        # calculate sentence embeddings for each word (node)
        sent_ids = [i_s for i_s, s in enumerate(data_sentences) if vocab[node] in s]

        sents_embds = [sentence_embeddings[sent_id] for sent_id in sent_ids]
        sents_avg_embd = np.average(sents_embds, axis=0)
        node_sentence_embeddings[node] = sents_avg_embd

        # calculate doc embeddings for each word (node)

        doc_ids = [i_d for i_d, doc in enumerate(data_docs) if vocab[node] in doc.split()]
        doc_embds = [doc_embeddings[doc_id] for doc_id in doc_ids]

        if len(doc_embds) == 1:
            node_doc_embeddings[node] = doc_embds[0]
        elif len(doc_embds) > 1:
            node_doc_embeddings[node] = np.average(doc_embds, axis=0)

    assert len(node_doc_embeddings) == len(nodes)
    assert len(node_sentence_embeddings) == len(nodes)

    return node_sentence_embeddings, node_doc_embeddings


def get_doc2vec_embeddings(processed_data: list, vocab: list, min_c: int, win: int, negative: int, hs: int,
                           seed: int, sample: float = 6e-5, alpha: float = 0.03, min_alpha: float = 0.0007,
                           epochs: int = 30, size: int = 300, ns_exponent: float = 0.75, dm=1, dbow_words=0):

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_data)]
    d2v_model = Doc2Vec(documents, min_count=min_c, window=win, vector_size=size, sample=sample, alpha=alpha,
                        min_alpha=min_alpha, negative=negative, ns_exponent=ns_exponent, seed=seed, compute_loss=True,
                        workers=1, epochs=epochs, sorted_vocab=1, hs=hs, dm=dm, dbow_words=dbow_words)

    # normalize vectors:
    d2v_model.init_sims(replace=True)

    # vocab_words and vocab_embeddings are sorted like vocab
    vocab_words = [w for w in vocab if w in d2v_model.wv.index2word]
    vocab_embeddings = [d2v_model.wv.vectors[d2v_model.wv.index2word.index(w)] for w in vocab_words]
    doc_embeddings = [d2v_model.docvecs[i] for i in range(len(processed_data))]

    return vocab_words, vocab_embeddings, doc_embeddings, d2v_model


def get_topic_vector(topic_embeddings: list):

    return np.average(topic_embeddings, axis=0)


if __name__ == "__main__":
    print("testing")
    print(api.info()['models'])
    # w2v_model = Word2Vec.load("w2v_node2vec")
    # word, word_embeddings, w2v_model_test = get_word_vectors([], "data/w2v_node2vec")
