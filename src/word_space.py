from src.visualizations import *
from src.vectorization import *
from src.clustering import *
from src.graphs import *


def get_w2v_vis_sign_words(data_processed: list, vocab: list, tokenized_docs: list, test_tokenized_docs: list,
                           x: list = None):
    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)
    w2v_params = {"min_c": 10, "win": 7, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42,
                  'ns_exponent': 0.75}

    y_c_v_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    y_dbs_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}

    test_y_c_v_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    test_y_npmi_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}

    y_topics = {'kmeans': [], 'agglomerative': [], 'hdbscan': []}

    words, word_embeddings, _ = get_word_vectors(data_processed, vocab, params=w2v_params)

    labels, probabilities = hdbscan_clustering(word_embeddings, min_cluster_size=6, do_dim_reduction=False)
    temp_cluster_words = [[] for _ in range(len(set(labels)) - 1)]
    temp_cluster_embeddings = [[] for _ in range(len(set(labels)) - 1)]
    for i, label in enumerate(labels):

        if label == -1:
            # noise
            continue
        temp_cluster_words[label].append([words[i], probabilities[i]])
        temp_cluster_embeddings[label].append(word_embeddings[i])

    hdbscan_clusters_words = []
    hdbscan_clusters_words_embeddings = []
    for i_c, c in enumerate(temp_cluster_words):
        c_sorted_indices = sorted(range(len(c)), key=lambda i_w: c[i_w][1], reverse=True)

        hdbscan_clusters_words.append([c[i][0] for i in c_sorted_indices[:30]])
        hdbscan_clusters_words_embeddings.append([temp_cluster_embeddings[i_c][i]
                                          for i in c_sorted_indices[:30]])

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "hdbscan"]:
            
            if cluster_type == "hdbscan":
                clusters_words = hdbscan_clusters_words
                clusters_words_embeddings = hdbscan_clusters_words_embeddings

            else:
                assert cluster_type in ["kmeans", "agglomerative"]
                if cluster_type == "kmeans":
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}
    
                clusters_words, clusters_words_embeddings = word_clusters(
                    data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                    params=clustering_params, clustering_weight_type=clustering_weight_type,
                    ranking_weight_type=ranking_weight_type,
                )

            print([len(c) for c in clusters_words])
            y_topics[cluster_type].append(clusters_words)

            # intrinsic
            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, clusters_words, cs_type='c_v')))
            dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
            cs_npmi = average_npmi_topics(tokenized_docs, clusters_words, len(clusters_words))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_dbs_model[cluster_type].append(dbs)

            # extrinsic
            test_y_c_v_model[cluster_type].append(float("{:.2f}".format(coherence_score(test_tokenized_docs,
                                                                                        clusters_words,
                                                                                        cs_type='c_v'))))
            test_y_npmi_model[cluster_type].append(average_npmi_topics(test_tokenized_docs, clusters_words,
                                                                       len(clusters_words)))

    # c_v coherence score - intrinsic
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/ws_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/ws_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # c_v coherence score - extrinsic
    ys = [l for l in test_y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/ws_extrinsic_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score - extrinsic
    ys = [l for l in test_y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/ws_extrinsic_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        assert len(y_c_v_model[m]) == len(y_npmi_model[m])
        assert len(y_npmi_model[m]) == len(y_dbs_model[m])
        assert len(y_dbs_model[m]) == len(y_topics[m])
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], test_y_c_v_model[m], test_y_npmi_model[m],
                         "visuals/ws_clusters_eval_" + str(m) + ".txt", dbs_scores=y_dbs_model[m])


def get_w2v_vis_topic_vec(data_processed: list, vocab: list, tokenized_docs: list, test_tokenized_docs: list,
                          x: list = None):
    clustering_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    # w2v_params = {"min_c": 50, "win": 15, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42}
    w2v_params = {"min_c": 10, "win": 7, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42,
                  'ns_exponent': 0.75}

    words, word_embeddings, _ = get_word_vectors(data_processed, vocab, params=w2v_params)

    y_c_v_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    y_dbs_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}

    test_y_c_v_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    test_y_npmi_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}

    y_topics = {'kmeans': [], 'agglomerative': [], 'hdbscan': []}

    labels, probabilities = hdbscan_clustering(word_embeddings, min_cluster_size=6, do_dim_reduction=False)

    temp_cluster_words = [[] for _ in range(len(set(labels)) - 1)]
    temp_cluster_embeddings = [[] for _ in range(len(set(labels)) - 1)]
    for i, label in enumerate(labels):

        if label == -1:
            # noise
            continue
        temp_cluster_words[label].append([words[i], probabilities[i]])
        temp_cluster_embeddings[label].append(word_embeddings[i])

    hdbscan_clusters_words = []
    hdbscan_clusters_words_embeddings = []
    for i_c, c in enumerate(temp_cluster_words):
        c_sorted_indices = sorted(range(len(c)), key=lambda i_w: c[i_w][1], reverse=True)

        hdbscan_clusters_words.append([c[i][0] for i in c_sorted_indices[:10]])
        hdbscan_clusters_words_embeddings.append([temp_cluster_embeddings[i_c][i]
                                          for i in c_sorted_indices[:10]])

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "hdbscan"]:

            if cluster_type == "hdbscan":
                clusters_words = hdbscan_clusters_words
                clusters_words_embeddings = hdbscan_clusters_words_embeddings

            else:
                assert cluster_type in ["kmeans", "agglomerative"]
                if cluster_type == "kmeans":
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                clusters_words, clusters_words_embeddings = word_clusters(
                    data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                    params=clustering_params, clustering_weight_type=clustering_weight_type,
                    ranking_weight_type=None)

            topic_vectors = [get_topic_vector(c) for c in clusters_words_embeddings]

            # get topics based on topic vectors
            topic_vector_cluster_words = []
            topic_vector_cluster_words_embeddings = []
            for t_vector in topic_vectors:
                sim_indices = get_most_similar_indices(t_vector, word_embeddings)

                topic_vector_cluster_words.append([words[i_w] for i_w in sim_indices])
                topic_vector_cluster_words_embeddings.append([word_embeddings[i_w] for i_w in sim_indices])

            y_topics[cluster_type].append(topic_vector_cluster_words)

            # intrinsic coherence
            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, topic_vector_cluster_words, cs_type='c_v')))
            dbs = float("{:.2f}".format(davies_bouldin_index(topic_vector_cluster_words_embeddings)))
            cs_npmi = average_npmi_topics(tokenized_docs, topic_vector_cluster_words, len(topic_vector_cluster_words))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_dbs_model[cluster_type].append(dbs)

            # extrinsic coherence
            test_y_c_v_model[cluster_type].append(float("{:.2f}".format(coherence_score(test_tokenized_docs,
                                                                                        clusters_words,
                                                                                        cs_type='c_v'))))
            test_y_npmi_model[cluster_type].append(average_npmi_topics(test_tokenized_docs, clusters_words,
                                                                       len(clusters_words)))

    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/ws_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # npmi coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/ws_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # c_v coherence score - extrinsic
    ys = [l for l in test_y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/ws_extrinsic_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score - extrinsic
    ys = [l for l in test_y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/ws_extrinsic_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], test_y_c_v_model[m], test_y_npmi_model[m],
                         "visuals/ws_clusters_eval_" + str(m) + ".txt", dbs_scores=y_dbs_model[m])


def get_graph_components(all_data_processed, vocab, tokenized_docs, test_tokenized_docs):
    n_words = len([w for d in all_data_processed for w in d])
    word_weights = get_word_weights(all_data_processed, vocab, n_words, weight_type='tf')

    w2v_params = {"min_c": 50, "win": 15, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42}

    vocab_words, vocab_embeddings, w2v_model = get_word_vectors(all_data_processed, vocab, params=w2v_params)

    y_topics = {1: [], 2: [], 3: []}
    y_c_v_clustering_type = {1: [], 2: [], 3: []}
    y_dbs_clustering_type = {1: [], 2: [], 3: []}
    y_npmi_clustering_type = {1: [], 2: [], 3: []}

    test_y_c_v_clustering_type = {1: [], 2: [], 3: []}
    test_y_npmi_clustering_type = {1: [], 2: [], 3: []}

    x = [x for x in range(50, 100, 10)] + [95]

    for sim in x:

        graph = create_networkx_graph(vocab_words, vocab_embeddings, similarity_threshold=0.8, percentile_cutoff=sim)

        components_all = apxa.k_components(graph)

        for k_component in [1, 2, 3]:

            print("k_component: " + str(k_component))
            components = components_all[k_component]

            corpus_clusters = []
            for comp in components:
                if len(comp) >= 6:
                    corpus_clusters.append(comp)

            cluster_words = [sorted(list(c), key=(lambda w: sort_words_by(graph, w, word_weights)),
                                    reverse=True) for c in corpus_clusters]

            if len(cluster_words) <= 2:
                print("here for: " + str(k_component) + ", " + str(sim))
                cs_c_v = -1000.0
                dbs = -1000.0
                cs_npmi = -1000.0
                cs_c_v_test = -1000.0
                cs_npmi_test = -1000.0
            else:

                cluster_embeddings = [[w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)] for w in words]
                                      for words in cluster_words]

                cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, cluster_words, cs_type='c_v')))
                dbs = float("{:.2f}".format(davies_bouldin_index(cluster_embeddings)))
                cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed, cluster_words,
                                                                    len(cluster_words))))

                # extrinsic
                cs_c_v_test = float("{:.2f}".format(coherence_score(test_tokenized_docs, cluster_words, cs_type='c_v')))
                cs_npmi_test = float("{:.2f}".format(average_npmi_topics(test_tokenized_docs, cluster_words,
                                                                         len(cluster_words))))

            y_c_v_clustering_type[k_component].append(cs_c_v)
            y_npmi_clustering_type[k_component].append(cs_npmi)
            y_dbs_clustering_type[k_component].append(dbs)

            y_topics[k_component].append(cluster_words)

            test_y_c_v_clustering_type[k_component].append(cs_c_v_test)
            test_y_npmi_clustering_type[k_component].append(cs_npmi_test)

    # c_v coherence score
    ys = [l for l in y_c_v_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Percentile", y_label="Coherence Score (c_v)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_v')
    fig.savefig("visuals/graph_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # npmi coherence score
    ys = [l for l in y_npmi_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Percentile", y_label="Coherence Score (NMPI)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_npmi')
    fig.savefig("visuals/graph_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Percentile", y_label="Davies–Bouldin index",
                          color_legends=["K=1", "K=2", "K=3"], type='dbs')
    fig.savefig("visuals/graph_dbi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # c_v coherence score - extrinsic
    ys = [l for l in test_y_c_v_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Percentile", y_label="Coherence Score (c_v)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_v')
    fig.savefig("visuals/extrinsic_graph_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score - extrinsic
    ys = [l for l in test_y_npmi_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Percentile", y_label="Coherence Score (NMPI)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_npmi')
    fig.savefig("visuals/extrinsic_graph_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_clustering_type[m], y_npmi_clustering_type[m],
                         test_y_c_v_clustering_type[m], test_y_npmi_clustering_type[m],
                         "visuals/graph_clusters_eval_" + str(m) + ".txt", dbs_scores=y_dbs_clustering_type[m])


def bert_visualization(all_data_processed: list, vocab: list, test_tokenized_docs: list, x: list = None):
    # normal_ll: "data/all_vocab_emb_dict_11_512.pickle"
    # normal_12: "all_vocab_emb_dict_12"

    # preprocessed_sentence_11: train_vocab_emb_dict_11_512_processed
    # preprocessed_sentence_12: train_vocab_emb_dict_12_512_processed

    # preprocessed_seg_11: train_vocab_emb_dict_11_512_processedSEG_words
    # preprocessed_seg_12: train_vocab_emb_dict_12_512_processedSEG_words
    with open("data/train_vocab_emb_dict_11_512_processedSEG_words.pickle", "rb") as f:
        all_vocab_emb_dict = pickle.load(f)

    all_vocab_emb_dict_words = all_vocab_emb_dict.keys()
    vocab_bert_embeddings = []
    new_vocab = []
    min_count = 80 # 100
    for word in vocab:

        if word in all_vocab_emb_dict_words:

            w_embeddings = all_vocab_emb_dict[word]

            if len(w_embeddings) >= min_count:
                new_vocab.append(word)
                vocab_bert_embeddings.append(np.average(w_embeddings, axis=0))
            else:
                continue
    assert all(len(embd) == 768 for embd in vocab_bert_embeddings)

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    words, word_embeddings = (new_vocab, vocab_bert_embeddings)

    y_c_v_model = {"kmeans": [], "agglomerative": [], 'hdbscan': []}
    y_dbs_model = {"kmeans": [], "agglomerative": [], 'hdbscan': []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], 'hdbscan': []}

    test_y_c_v_model = {"kmeans": [], "agglomerative": [], 'hdbscan': []}
    test_y_npmi_model = {"kmeans": [], "agglomerative": [], 'hdbscan': []}

    y_topics = {'kmeans': [], 'agglomerative': [], 'hdbscan': []}

    labels, probabilities = hdbscan_clustering(word_embeddings, min_cluster_size=6, do_dim_reduction=True)

    temp_cluster_words = [[] for _ in range(len(set(labels)) - 1)]
    temp_cluster_embeddings = [[] for _ in range(len(set(labels)) - 1)]
    for i, label in enumerate(labels):

        if label == -1:
            # noise
            continue
        temp_cluster_words[label].append([words[i], probabilities[i]])
        temp_cluster_embeddings[label].append(word_embeddings[i])

    hdbscan_clusters_words = []
    hdbscan_clusters_words_embeddings = []
    for i_c, c in enumerate(temp_cluster_words):
        c_sorted_indices = sorted(range(len(c)), key=lambda i_w: c[i_w][1], reverse=True)

        hdbscan_clusters_words.append([c[i][0] for i in c_sorted_indices[:30]])
        hdbscan_clusters_words_embeddings.append([temp_cluster_embeddings[i_c][i]
                                          for i in c_sorted_indices[:30]])
    print("HDBSCAN len: " + str(len(hdbscan_clusters_words)))
    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "hdbscan"]:

            if cluster_type == "hdbscan":

                # clusters_words = hdbscan_clusters_words
                clusters_words_embeddings = hdbscan_clusters_words_embeddings

            else:

                if cluster_type in ["kmeans", "nmf"]:
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                clusters_words, clusters_words_embeddings = word_clusters(
                    all_data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                    params=clustering_params, clustering_weight_type='tf',
                    ranking_weight_type=None
                )

            topic_vectors = [get_topic_vector(c) for c in clusters_words_embeddings]

            # get topics based on topic vectors
            topic_vector_cluster_words = []
            topic_vector_cluster_words_embeddings = []
            for t_vector in topic_vectors:
                sim_indices = get_most_similar_indices(t_vector, word_embeddings, n_most_similar=30)

                topic_vector_cluster_words.append([words[i_w] for i_w in sim_indices])
                topic_vector_cluster_words_embeddings.append([word_embeddings[i_w] for i_w in sim_indices])

            if len(topic_vector_cluster_words) <= 2:
                cs_c_v = -1000.0
                dbs = -1000.0
                cs_npmi = -1000.0

                test_cs_c_v = -1000.0
                test_cs_npmi = -1000.0

            else:

                cs_c_v = float("{:.2f}".format(coherence_score(all_data_processed, topic_vector_cluster_words,
                                                               cs_type='c_v')))
                dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
                cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed,
                                                                    topic_vector_cluster_words,
                                                                    len(topic_vector_cluster_words))))

                test_cs_c_v = float(
                    "{:.2f}".format(coherence_score(test_tokenized_docs, topic_vector_cluster_words, cs_type='c_v')))
                test_cs_npmi = float("{:.2f}".format(average_npmi_topics(test_tokenized_docs,
                                                                         topic_vector_cluster_words,
                                                                         len(topic_vector_cluster_words))))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_dbs_model[cluster_type].append(dbs)

            test_y_c_v_model[cluster_type].append(test_cs_c_v)
            test_y_npmi_model[cluster_type].append(test_cs_npmi)

            y_topics[cluster_type].append(topic_vector_cluster_words)

    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/w_space_c_v_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

    # c_npmi coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/w_space_c_npmi_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='dbs')
    fig.savefig("visuals/w_space_dbi_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

    # c_v coherence score - extrinsic
    ys = [l for l in test_y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_v')
    fig.savefig("visuals/extrinsic_graph_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score - extrinsic
    ys = [l for l in test_y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_npmi')
    fig.savefig("visuals/extrinsic_graph_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(topics_list=y_topics[m], c_v_scores=y_c_v_model[m], nmpi_scores=y_npmi_model[m],
                         test_c_v_scores=test_y_c_v_model[m],
                         test_nmpi_scores=test_y_npmi_model[m],
                         filename="visuals/w_space_clusters_eval_" + str(m) + ".txt",
                         dbs_scores=y_dbs_model[m])