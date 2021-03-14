from src.vectorization import *
from src.clustering import *
from src.graphs import *
from src.misc import save_model_scores
import pickle
from networkx.algorithms import approximation as apxa
from pathlib import Path

w2v_params = {"min_c": 10, "win": 7, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42,
              'ns_exponent': 0.75}


def word2vec_topic_model(data_processed: list, vocab: list, tokenized_docs: list, test_tokenized_docs: list,
                         data_set_name: str, x: list = None, topic_vector_flag: bool = False):
    """
    word2vec_topic_model performs topic modeling in word space using Word2Vec embeddings.
    The function produces a range of files that list the resulting topics and visualize the model's performance.

    :param data_processed: preprocessed data set used to calculated word embeddings
    :param vocab: vocabulary of the preprocessed data set
    :param tokenized_docs: tokenized version of the training data set
    :param test_tokenized_docs: tokenized version of the test data set
    :param data_set_name: preprocessed data set used
    :param x: list of number of topics to iterate over, default: list(range(2, 22, 2))
    :param topic_vector_flag: flag used to switch between TVS model and RRW and, default: False (RRW model)

    """

    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    assert data_set_name in ["JN", "FP"]

    y_c_v_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    y_dbs_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    y_npmi_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    test_y_c_v_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    test_y_npmi_model = {"K-Means": [], "Agglomerative": [], "HDBSCAN": []}
    y_topics = {'K-Means': [], 'Agglomerative': [], 'HDBSCAN': []}

    # words, word_embeddings, _ = get_word_vectors(data_processed, vocab, params=w2v_params)
    w2v_model_file = "w2v_model-" + data_set_name + ".pickle"
    if Path("data/" + w2v_model_file).is_file():

        print("using pre-calculated w2v model")
        with open("data/" + w2v_model_file, "rb") as myFile:
            w2v_model = pickle.load(myFile)

            words = [w for w in vocab if w in w2v_model.wv.index2word]
            word_embeddings = [w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)] for w in words]

    else:
        words, word_embeddings, w2v_model = get_word_vectors(data_processed, vocab, params=w2v_params)

        with open("data/"+ w2v_model_file, "wb") as myFile:
            pickle.dump(w2v_model, myFile)

    _, _, hdbscan_clusters_words, hdbscan_clusters_words_embeddings = hdbscan_clustering(words, word_embeddings,
                                                                                         min_cluster_size=6,
                                                                                         do_dim_reduction=False)
    for k in x:

        for cluster_type in ["K-Means", "Agglomerative", "HDBSCAN"]:
            
            if cluster_type == "HDBSCAN":
                clusters_words = hdbscan_clusters_words
                clusters_words_embeddings = hdbscan_clusters_words_embeddings

            else:
                assert cluster_type in ["K-Means", "Agglomerative"]
                if cluster_type == "K-Means":
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                if topic_vector_flag:
                    ranking_weight = None
                else:
                    ranking_weight = ranking_weight_type

                clusters_words, clusters_words_embeddings = get_word_clusters(
                    data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                    params=clustering_params, clustering_weight_type=clustering_weight_type,
                    ranking_weight_type=ranking_weight,
                )

            if topic_vector_flag:
                topic_vectors = [get_topic_vector(c) for c in clusters_words_embeddings]

                # get topics based on topic vectors
                topic_vector_cluster_words = []
                topic_vector_cluster_words_embeddings = []
                for t_vector in topic_vectors:
                    sim_indices = get_nearest_indices(t_vector, word_embeddings)

                    topic_vector_cluster_words.append([words[i_w] for i_w in sim_indices])
                    topic_vector_cluster_words_embeddings.append([word_embeddings[i_w] for i_w in sim_indices])
                clusters_words = topic_vector_cluster_words

            y_topics[cluster_type].append(clusters_words)

            # intrinsic evaluation scores
            y_c_v_model[cluster_type].append(c_v_coherence_score(tokenized_docs, clusters_words))
            y_npmi_model[cluster_type].append(npmi_coherence_score(tokenized_docs, clusters_words, len(clusters_words)))
            y_dbs_model[cluster_type].append(davies_bouldin_index(clusters_words_embeddings))

            # extrinsic evaluation scores
            test_y_c_v_model[cluster_type].append(c_v_coherence_score(test_tokenized_docs, clusters_words))
            test_y_npmi_model[cluster_type].append(npmi_coherence_score(test_tokenized_docs, clusters_words,
                                                                        len(clusters_words)))
    if topic_vector_flag:
        filename_prefix = "TVS"
    else:
        filename_prefix = "RRW"

    save_model_scores(x_values=x, models=list(y_topics.keys()), model_topics=y_topics, model_c_v_scores=y_c_v_model,
                      model_npmi_scores=y_npmi_model, model_c_v_test_scores=test_y_c_v_model,
                      model_npmi_test_scores=test_y_npmi_model, filename_prefix=filename_prefix,
                      model_dbs_scores=y_dbs_model)


def k_components_model(data_processed: list, vocab: list, tokenized_docs: list, test_tokenized_docs: list,
                       data_set_name: str):
    """
    k_components_model is used to perform topic model on the word embedding graph using k-components algorithm.
    This function uses the k-components approximation function from the Networkx library

    :param data_processed: preprocessed data set used to calculated word embeddings
    :param vocab: vocabulary of the preprocessed data set
    :param tokenized_docs: tokenized version of the training data set
    :param test_tokenized_docs: tokenized version of the test data set
    :param data_set_name: name of the preprocessed data set used
    :return:
    """
    n_words = len([w for d in data_processed for w in d])
    word_weights = get_word_weights(data_processed, vocab, n_words, weight_type='tf')

    w2v_model_file = "w2v_model-k_components-" + data_set_name + ".pickle"
    if Path("data/" + w2v_model_file).is_file():

        print("using pre-calculated w2v model")
        with open("data/" + w2v_model_file, "rb") as myFile:
            w2v_model = pickle.load(myFile)

            vocab_words = [w for w in vocab if w in w2v_model.wv.index2word]
            vocab_embeddings = [w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)]
                                for w in vocab_words]

    else:
        w2v_params_k_components = {"min_c": 50, "win": 15, "negative": 0, "sample": 1e-5,
                                   "hs": 1, "epochs": 400, "sg": 1, 'seed': 42}
        vocab_words, vocab_embeddings, w2v_model = get_word_vectors(data_processed, vocab,
                                                                    params=w2v_params_k_components)

        with open("data/" + w2v_model_file, "wb") as myFile:
            pickle.dump(w2v_model, myFile)

    y_topics = {"K=1": [], "K=2": [], "K=3": []}
    y_c_v_model = {"K=1": [], "K=2": [], "K=3": []}
    y_dbs_model = {"K=1": [], "K=2": [], "K=3": []}
    y_npmi_model = {"K=1": [], "K=2": [], "K=3": []}

    test_y_c_v_model = {"K=1": [], "K=2": [], "K=3": []}
    test_y_npmi_model = {"K=1": [], "K=2": [], "K=3": []}

    x = [x for x in range(50, 100, 10)] + [95]

    for sim in x:

        graph = create_networkx_graph(vocab_words, vocab_embeddings, similarity_threshold=0.8, percentile_cutoff=sim)
        components_all = apxa.k_components(graph)

        for k_component in ["K=1", "K=2", "K=3"]:

            temp_k_dict = {"K=1": 1, "K=2": 2, "K=3": 3}
            components = components_all[temp_k_dict[k_component]]

            corpus_clusters = []
            for comp in components:
                if len(comp) >= 6:
                    corpus_clusters.append(comp)

            cluster_words = [sorted(list(c), key=(lambda w: sort_words_by(graph, w, word_weights)),
                                    reverse=True) for c in corpus_clusters]

            if len(cluster_words) <= 2:
                # topic model did not find enough topics
                # -1000.0 is the NaN value used in the charts, these values will not be shown in the charts
                cs_c_v = -1000.0
                dbs = -1000.0
                cs_npmi = -1000.0
                cs_c_v_test = -1000.0
                cs_npmi_test = -1000.0
            else:

                cluster_embeddings = [[w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)] for w in words]
                                      for words in cluster_words]

                # intrinsic evaluation
                cs_c_v = c_v_coherence_score(tokenized_docs, cluster_words)
                dbs = davies_bouldin_index(cluster_embeddings)
                cs_npmi = npmi_coherence_score(data_processed, cluster_words, len(cluster_words))

                # extrinsic evaluation
                cs_c_v_test = c_v_coherence_score(test_tokenized_docs, cluster_words)
                cs_npmi_test = npmi_coherence_score(test_tokenized_docs, cluster_words, len(cluster_words))

            y_topics[k_component].append(cluster_words)

            y_c_v_model[k_component].append(cs_c_v)
            y_npmi_model[k_component].append(cs_npmi)
            y_dbs_model[k_component].append(dbs)

            test_y_c_v_model[k_component].append(cs_c_v_test)
            test_y_npmi_model[k_component].append(cs_npmi_test)

    save_model_scores(x_values=x, models=list(y_topics.keys()), model_topics=y_topics, model_c_v_scores=y_c_v_model,
                      model_npmi_scores=y_npmi_model, model_c_v_test_scores=test_y_c_v_model,
                      model_npmi_test_scores=test_y_npmi_model, filename_prefix='k-components', 
                      model_dbs_scores=y_dbs_model)


def bert_topic_model(bert_embedding_type: str, all_data_processed: list, vocab: list, test_tokenized_docs: list,
                     x: list = None):
    """
    bert_topic_model fetched the predefined bert embeddings, clusters them and performs

    :param bert_embedding_type:
    :param all_data_processed:
    :param vocab:
    :param test_tokenized_docs:
    :param x: list of number of topics, default: list(range(2, 22, 2))
    """
    bert_file_names = {
        'normal_ll': "all_vocab_emb_dict_11",
        'normal_12': "all_vocab_emb_dict_12",
        'preprocessed_sentence_11': "train_vocab_emb_dict_11_512_processed",
        'preprocessed_sentence_12': "train_vocab_emb_dict_12_512_processed",
        'preprocessed_seg_11': "train_vocab_emb_dict_11_512_processedSEG_words",
        'preprocessed_seg_12': "train_vocab_emb_dict_12_512_processedSEG_words"

    }

    assert bert_embedding_type in bert_file_names.keys()
    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    # train_vocab_emb_dict_11_512_processedSEG_words
    with open("data/" + bert_file_names[bert_embedding_type] + ".pickle", "rb") as f:
        all_vocab_emb_dict = pickle.load(f)

    all_vocab_emb_dict_words = all_vocab_emb_dict.keys()
    vocab_bert_embeddings = []
    new_vocab = []
    min_count = 80  # 100
    for word in vocab:

        if word in all_vocab_emb_dict_words:

            w_embeddings = all_vocab_emb_dict[word]

            if len(w_embeddings) >= min_count:
                new_vocab.append(word)
                vocab_bert_embeddings.append(np.average(w_embeddings, axis=0))
            else:
                continue
    assert all(len(embd) == 768 for embd in vocab_bert_embeddings)

    words, word_embeddings = (new_vocab, vocab_bert_embeddings)

    y_c_v_model = {"K-Means": [], "Agglomerative": [], 'HDBSCAN': []}
    y_dbs_model = {"K-Means": [], "Agglomerative": [], 'HDBSCAN': []}
    y_npmi_model = {"K-Means": [], "Agglomerative": [], 'HDBSCAN': []}

    test_y_c_v_model = {"K-Means": [], "Agglomerative": [], 'HDBSCAN': []}
    test_y_npmi_model = {"K-Means": [], "Agglomerative": [], 'HDBSCAN': []}

    y_topics = {'K-Means': [], 'Agglomerative': [], 'HDBSCAN': []}

    _, _, hdbscan_clusters_words, hdbscan_clusters_words_embeddings = hdbscan_clustering(words, word_embeddings,
                                                                                         min_cluster_size=6,
                                                                                         do_dim_reduction=False,
                                                                                         n_words=30)
    print("HDBSCAN len: " + str(len(hdbscan_clusters_words)))
    for k in x:

        for cluster_type in ["K-Means", "Agglomerative", "HDBSCAN"]:

            if cluster_type == "HDBSCAN":

                # clusters_words = HDBSCAN_clusters_words
                clusters_words_embeddings = hdbscan_clusters_words_embeddings

            else:

                if cluster_type in ["K-Means"]:
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                clusters_words, clusters_words_embeddings = get_word_clusters(
                    all_data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                    params=clustering_params, clustering_weight_type='tf',
                    ranking_weight_type=None
                )

            topic_vectors = [get_topic_vector(c) for c in clusters_words_embeddings]

            # get topics based on topic vectors
            topic_vector_cluster_words = []
            topic_vector_cluster_words_embeddings = []
            for t_vector in topic_vectors:
                sim_indices = get_nearest_indices(t_vector, word_embeddings, n_nearest=30)

                topic_vector_cluster_words.append([words[i_w] for i_w in sim_indices])
                topic_vector_cluster_words_embeddings.append([word_embeddings[i_w] for i_w in sim_indices])

            if len(topic_vector_cluster_words) <= 2:
                cs_c_v = -1000.0
                dbs = -1000.0
                cs_npmi = -1000.0

                test_cs_c_v = -1000.0
                test_cs_npmi = -1000.0

            else:

                cs_c_v = c_v_coherence_score(all_data_processed, topic_vector_cluster_words)
                dbs = davies_bouldin_index(clusters_words_embeddings)
                cs_npmi = npmi_coherence_score(all_data_processed, topic_vector_cluster_words, 
                                               len(topic_vector_cluster_words))

                test_cs_c_v = c_v_coherence_score(test_tokenized_docs, topic_vector_cluster_words)
                test_cs_npmi = npmi_coherence_score(test_tokenized_docs, topic_vector_cluster_words,
                                                    len(topic_vector_cluster_words))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_dbs_model[cluster_type].append(dbs)

            test_y_c_v_model[cluster_type].append(test_cs_c_v)
            test_y_npmi_model[cluster_type].append(test_cs_npmi)

            y_topics[cluster_type].append(topic_vector_cluster_words)

    save_model_scores(x_values=x, models=list(y_topics.keys()), model_topics=y_topics, model_c_v_scores=y_c_v_model,
                      model_npmi_scores=y_npmi_model, model_c_v_test_scores=test_y_c_v_model,
                      model_npmi_test_scores=test_y_npmi_model, filename_prefix='BERT',
                      model_dbs_scores=y_dbs_model)
