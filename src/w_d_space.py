from src.model import *
from src.evaluation import *
from src.visualizations import *
from src.vectorization import *
from src.clustering import *
from src.graphs import *
# from top2vec import *
from sklearn.preprocessing import normalize


def w_d_clustering(all_data_processed: list, vocab: list, tokenized_docs: list, doc_labels_true: list,
                   doc_embedding_type="w2v_avg", x: list = None):

    # main extrinsic evaluation metric: ARI
    # https://stats.stackexchange.com/questions/381223/evaluation-of-clustering-method

    true_topic_amount = len(set(doc_labels_true))

    if x is None:
        x = list(range(2, 22, 2))
        assert true_topic_amount in x

    else:
        assert isinstance(x, list)

    min_cluster_size = 7
    if doc_embedding_type in ["w2v_avg", "w2v_sum"]:
        params = {"min_c": 10, "win": 7, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1,
                      'seed': 42, 'ns_exponent': 0.75}
        min_cluster_size = 7

    elif doc_embedding_type == "doc2vec":
        params = {"min_c": 10, "win": 5, "negative": 30, "sample": 1e-5, "hs": 0, "epochs": 400, 'seed': 42,
                      'ns_exponent': 0.75, "dm": 0, "dbow_words": 1}
        min_cluster_size = 8

    doc_data, short_true_labels, doc_embeddings, vocab_words, vocab_embeddings = get_doc_embeddings(
        all_data_processed, doc_labels_true, vocab, doc_embedding_type, params)

    y_topics = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}
    y_c_v_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}

    labels, probabilities = hdbscan_clustering(doc_embeddings, min_cluster_size=min_cluster_size, do_dim_reduction=False)

    print("HDBSCAN #labels: " + str(set(labels)))
    hdbscan_clusters_docs_embeddings = [[] for _ in range(len(set(labels)) - 1)]
    hdbscan_clusters_docs = [[] for _ in range(len(set(labels)) - 1)]
    hdbscan_labels_predict = []
    hdbscan_true_labels = []
    for i, label in enumerate(labels):

        if label == -1:
            # noise
            continue
        hdbscan_clusters_docs[label].append(doc_data[i])
        hdbscan_clusters_docs_embeddings[label].append(doc_embeddings[i])
        hdbscan_true_labels.append(short_true_labels[i])
        hdbscan_labels_predict.append(label)

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "HDBSCAN"]:

            if cluster_type == "HDBSCAN":
                clusters_docs = hdbscan_clusters_docs
                clusters_docs_embeddings = hdbscan_clusters_docs_embeddings
                labels_predict = hdbscan_labels_predict
                true_labels = hdbscan_true_labels

            else:
                if cluster_type in ["kmeans", "nmf"]:
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                # clustering document via document embeddings
                clusters_docs, clusters_docs_embeddings, labels_predict, doc_data = document_clustering(
                    doc_data, doc_embeddings, vocab, cluster_type, params=clustering_params)

                true_labels = short_true_labels

            topic_embeddings = []
            topics_words = []
            topics_words_embeddings = []
            for docs_embeddings in clusters_docs_embeddings:
                t_embedding = np.average(docs_embeddings, axis=0)

                t_embedding_sim_matrix = cosine_similarity(t_embedding.reshape(1, -1), vocab_embeddings)[0]
                most_sim_ids = np.argsort(t_embedding_sim_matrix, axis=None)[:: -1]

                t_words = [vocab_words[i] for i in most_sim_ids[:10]]
                t_words_embeddings = [vocab_embeddings[i] for i in most_sim_ids[:10]]

                topics_words.append(t_words)
                topics_words_embeddings.append(t_words_embeddings)
                topic_embeddings.append(t_embedding)

            cs_c_v = float("{:.2f}".format(coherence_score(all_data_processed, topics_words, cs_type='c_v')))
            print(cluster_type)
            print("c_v: " + str(cs_c_v))
            cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed, topics_words, len(topics_words))))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_topics[cluster_type].append(topics_words)

            if k == true_topic_amount:
                vis_classification_score(cluster_type, true_labels, labels_predict, topics_words,
                                         "visuals/w_d_space_classification_scores_" + str(cluster_type) + ".txt")
    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/w_d_space_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # npmi coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (npmi)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/w_d_space_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m],
                         "visuals/w_d_space_clusters_eval_" + str(m) + ".txt")


def test_clustering(all_data_processed: list, vocab: list, tokenized_docs: list, doc_labels_true: list,
                   doc_embedding_type="w2v_avg", x: list = None):

    # main extrinsic evaluation metric: ARI
    # https://stats.stackexchange.com/questions/381223/evaluation-of-clustering-method

    # clustering_weight_type = 'tf'
    # ranking_weight_type = 'tf'
    true_topic_amount = len(set(doc_labels_true))

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    d2v_params = {"size": 300, "min_c": 15, "win": 15, "sample": 1e-5, "negative": 0, "hs": 1, "epochs": 400,
                  "seed": 42}

    doc_data, short_true_labels, doc_embeddings, vocab_words, vocab_embeddings = get_doc_embeddings(
        all_data_processed, doc_labels_true, vocab, doc_embedding_type, d2v_params)

    y_topics = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}
    y_c_v_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "HDBSCAN"]:

            if cluster_type == "HDBSCAN":
                labels, probabilities = hdbscan_clustering(doc_embeddings, min_cluster_size=12, do_dim_reduction=False,
                                                           do_cosine_similarity=False, )
                print("HDBSCAN #labels: " + str(set(labels)))
                hdbscan_clusters_docs = [[] for _ in range(len(set(labels)) - 1)]
                hdbscan_clusters_docs_embeddings = [[] for _ in range(len(set(labels)) - 1)]
                hdbscan_labels_predict = []
                hdbscan_true_labels = []
                for i, label in enumerate(labels):

                    if label == -1:
                        # noise
                        continue
                    hdbscan_clusters_docs[label].append([doc_data[i], probabilities[i]])
                    hdbscan_clusters_docs_embeddings[label].append(doc_embeddings[i])
                    hdbscan_labels_predict.append(label)
                    hdbscan_true_labels.append(short_true_labels[i])

                true_labels = hdbscan_true_labels
                clusters_docs_embeddings = hdbscan_clusters_docs_embeddings
                labels_predict = hdbscan_labels_predict
            else:
                if cluster_type in ["kmeans", "nmf"]:
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                # clustering document via document embeddings
                clusters_docs, clusters_docs_embeddings, labels_predict, doc_data = document_clustering(
                    doc_data, doc_embeddings, vocab, cluster_type, params=clustering_params)
                true_labels = short_true_labels
            topic_embeddings = []
            topics_words = []
            topics_words_embeddings = []
            for docs_embeddings in clusters_docs_embeddings:
                t_embedding = np.average(docs_embeddings, axis=0)

                t_embedding_sim_matrix = cosine_similarity(t_embedding.reshape(1, -1), vocab_embeddings)[0]
                most_sim_ids = np.argsort(t_embedding_sim_matrix, axis=None)[:: -1]

                t_words = [vocab_words[i] for i in most_sim_ids[:10]]
                t_words_embeddings = [vocab_embeddings[i] for i in most_sim_ids[:10]]

                topics_words.append(t_words)
                topics_words_embeddings.append(t_words_embeddings)
                topic_embeddings.append(t_embedding)

            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, topics_words, cs_type='c_v')))
            cs_npmi = float("{:.2f}".format(average_npmi_topics(tokenized_docs, topics_words, len(topics_words))))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_topics[cluster_type].append(topics_words)

            if k == true_topic_amount:
                vis_classification_score(cluster_type, true_labels, labels_predict, topics_words,
                                         "visuals/classification_scores_" + str(cluster_type) + ".txt")

    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/c_v_doc_vs_k.pdf", bbox_inches='tight', transparent=True)

    # npmi coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (npmi)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/npmi_doc_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], "visuals/d2v_clusters_eval_" + str(m) + ".txt")


def bert_visualization(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):

    with open("data/all_vocab_emb_dict_snd_last_bare.pickle", "rb") as f:
        all_vocab_emb_dict = pickle.load(f)

    all_vocab_emb_dict_words = all_vocab_emb_dict.keys()
    vocab_bert_embeddings = []
    new_vocab = []
    for word in vocab:

        if word in all_vocab_emb_dict_words:

            w_embeddings = all_vocab_emb_dict[word]

            if len(w_embeddings) >= 100:
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

    y_c_v_model = {"kmeans": [], "agglomerative": [], "nmf": [], 'hdbscan': []}
    y_dbs_model = {"kmeans": [], "agglomerative": [], "nmf": [], 'hdbscan': []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], "nmf": [], 'hdbscan': []}

    y_topics = {'kmeans': [], 'agglomerative': [], "nmf": [], 'hdbscan': []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "nmf", "hdbscan"]:

            if cluster_type == "hdbscan":
                labels, probabilities = hdbscan_clustering(word_embeddings, min_cluster_size=6, do_dim_reduction=True)

                temp_cluster_words = [[] for _ in range(len(set(labels)) - 1)]
                temp_cluster_embeddings = [[] for _ in range(len(set(labels)) - 1)]
                for i, label in enumerate(labels):

                    if label == -1:
                        # noise
                        continue
                    temp_cluster_words[label].append([words[i], probabilities[i]])
                    temp_cluster_embeddings[label].append(word_embeddings[i])

                clusters_words = []
                clusters_words_embeddings = []
                for i_c, c in enumerate(temp_cluster_words):
                    c_sorted_indices = sorted(range(len(c)), key=lambda i_w: c[i_w][1], reverse=True)

                    clusters_words.append([c[i][0] for i in c_sorted_indices[:10]])
                    clusters_words_embeddings.append([temp_cluster_embeddings[i_c][i]
                                                      for i in c_sorted_indices[:10]])

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
                sim_indices = get_most_similar_indices(t_vector, word_embeddings, n_most_similar=10)

                topic_vector_cluster_words.append([words[i_w] for i_w in sim_indices])
                topic_vector_cluster_words_embeddings.append([word_embeddings[i_w] for i_w in sim_indices])

            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, topic_vector_cluster_words, cs_type='c_v')))
            dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
            cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed,
                                                                topic_vector_cluster_words,
                                                                len(topic_vector_cluster_words))))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_dbs_model[cluster_type].append(dbs)

            y_topics[cluster_type].append(topic_vector_cluster_words)

            for m in list(y_topics.keys()):
                vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m],
                                 "visuals/w_d_space_clusters_eval_" + str(m) + ".txt",
                                 dbs_scores=y_dbs_model[m])

    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "NMF", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/w_d_space_c_v_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

    # c_npmi coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="NPMI",
                          color_legends=["K-Means", "Agglomerative", "NMF", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/w_d_space_c_npmi_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative", "NMF", "HDBSCAN"], type='dbs')
    fig.savefig("visuals/w_d_space_dbi_bert_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], "visuals/ws_clusters_eval_" + str(m) + ".txt",
                         dbs_scores=y_dbs_model[m])


def w_d_get_graph_components(all_data_processed, vocab, tokenized_docs, doc_labels_true, doc_embedding_type="w2v_avg"):
    n_words = len([w for d in all_data_processed for w in d])
    word_weights = get_word_weights(all_data_processed, vocab, n_words, weight_type='tf')

    d2v_params = {"size": 300, "min_c": 15, "win": 15, "sample": 1e-5, "negative": 0, "hs": 1, "epochs": 400,
                  "seed": 42}
    doc_data, short_true_labels, doc_embeddings, vocab_words, vocab_embeddings = get_doc_embeddings(
        all_data_processed, doc_labels_true, vocab, doc_embedding_type, d2v_params)

    doc_indices = list(range(len(doc_data)))

    y_topics = {1: [], 2: [], 3: []}
    y_c_v_clustering_type = {1: [], 2: [], 3: []}
    y_dbs_clustering_type = {1: [], 2: [], 3: []}
    y_npmi_clustering_type = {1: [], 2: [], 3: []}

    x = [x/100 for x in range(70, 100, 10)] + [.95]

    for sim in x:
        print("Creating graph!")
        graph = create_networkx_graph(doc_indices, doc_embeddings, similarity_threshold=sim, percentile_cutoff=80)
        print("Created graph!")
        components_all = apxa.k_components(graph)

        for k_component in [1, 2, 3]:

            print("k_component: " + str(k_component))
            components = components_all[k_component]

            corpus_clusters = []
            for comp in components:
                if len(comp) >= 6:
                    corpus_clusters.append(comp)

            cluster_docs = [sorted(list(c), key=(lambda w: sort_words_by(graph, w, word_weights)), reverse=True)
                            for c in corpus_clusters]

            if len(cluster_docs) <= 2:
                print("here for: " + str(k_component) + ", " + str(sim))
                cs_c_v = -1000.0
                dbs = -1000.0
                cs_npmi = -1000.0

            else:

                cluster_embeddings = [[doc_embeddings[i] for i in docs] for docs in cluster_docs]

                topic_vectors = [get_topic_vector(c) for c in cluster_embeddings]

                # get topics based on topic vectors
                topic_vector_cluster_words = []
                topic_vector_cluster_words_embeddings = []
                for t_vector in topic_vectors:
                    sim_indices = get_most_similar_indices(t_vector, vocab_embeddings, n_most_similar=10)

                    topic_vector_cluster_words.append([vocab_words[i_w] for i_w in sim_indices])
                    topic_vector_cluster_words_embeddings.append([vocab_embeddings[i_w] for i_w in sim_indices])

                cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, topic_vector_cluster_words,
                                                               cs_type='c_v')))
                dbs = float("{:.2f}".format(davies_bouldin_index(topic_vector_cluster_words_embeddings)))
                cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed, topic_vector_cluster_words,
                                                                    len(topic_vector_cluster_words))))

            y_c_v_clustering_type[k_component].append(cs_c_v)
            y_npmi_clustering_type[k_component].append(cs_npmi)
            y_dbs_clustering_type[k_component].append(dbs)

            y_topics[k_component].append(cluster_docs)

    # c_v coherence score
    ys = [l for l in y_c_v_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="Coherence Score (c_v)",
                          color_legends=["K=1", "K=2", "K=3"], type='c_v')
    fig.savefig("visuals/w_d_space_graph_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # npmi coherence score
    ys = [l for l in y_npmi_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="NPMI",
                          color_legends=["K=1", "K=2", "K=3"], type='c_npmi')
    fig.savefig("visuals/w_d_space_graph_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Similarity Threshold", y_label="Davies–Bouldin index",
                          color_legends=["K=1", "K=2", "K=3"], type='dbs')
    fig.savefig("visuals/w_d_space_graph_dbi_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_clustering_type[m], y_npmi_clustering_type[m],
                         "visuals/w_d_space_graph_clusters_eval_" + str(m) + ".txt",
                         dbs_scores=y_dbs_clustering_type[m])


def alberta_visualization(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):

    with open("data/all_vocab_emb_dict_alberta_11.pickle", "rb") as f:
        all_vocab_emb_dict = pickle.load(f)

    all_vocab_emb_dict_words = all_vocab_emb_dict.keys()
    vocab_bert_embeddings = []
    new_vocab = []
    for word in vocab:

        if word in all_vocab_emb_dict_words:

            w_embeddings = all_vocab_emb_dict[word]

            if len(w_embeddings) >= 100:
                new_vocab.append(word)
                vocab_bert_embeddings.append(np.average(w_embeddings, axis=0))
            else:
                continue
    assert all(len(embd) == 768 for embd in vocab_bert_embeddings)

    print("old vocab len: " + str(len(vocab)))
    print("new vocab len: " + str(len(new_vocab)))

    vocab = new_vocab
    vocab_embeddings = vocab_bert_embeddings

    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    words, word_embeddings = (vocab, vocab_embeddings)

    k_10_c_v = {"kmeans": 0, "agglomerative": 0, "nmf": 0}
    best_c_v = {"kmeans": 0, "agglomerative": 0, "nmf": 0}
    k_10_topics = {"kmeans": None, "agglomerative": None, "nmf": None}
    best_c_v_topics = {"kmeans": None, "agglomerative": None, "nmf": None}

    worst_c_v = {"kmeans": 1, "agglomerative": 1, "nmf": 1}
    worst_c_v_topics = {"kmeans": None, "agglomerative": None, "nmf": None}

    y_c_v_clustering_type = {"kmeans": [], "agglomerative": [], "nmf": []}
    y_dbs_clustering_type = {"kmeans": [], "agglomerative": [], "nmf": []}
    y_u_mass_clustering_type = {"kmeans": [], "agglomerative": [], "nmf": []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "nmf"]:

            if cluster_type in ["kmeans", "nmf"]:
                clustering_params = {'n_clusters': k, 'random_state': 42, }
            else:
                clustering_params = {'n_clusters': k}

            clusters_words, clusters_words_embeddings = word_clusters(
                all_data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
                params=clustering_params, clustering_weight_type=clustering_weight_type,
                ranking_weight_type=ranking_weight_type
            )

            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, clusters_words, cs_type='c_v')))
            dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
            cs_npmi = float("{:.2f}".format(average_npmi_topics(all_data_processed,
                                                                clusters_words, len(clusters_words))))

            y_c_v_clustering_type[cluster_type].append(cs_c_v)
            y_u_mass_clustering_type[cluster_type].append(cs_npmi)
            y_dbs_clustering_type[cluster_type].append(dbs)

            if cs_c_v > best_c_v[cluster_type]:
                best_c_v[cluster_type] = cs_c_v
                best_c_v_topics[cluster_type] = clusters_words

            if cs_c_v < worst_c_v[cluster_type]:
                worst_c_v[cluster_type] = cs_c_v
                worst_c_v_topics[cluster_type] = clusters_words

            if k == 10:
                k_10_c_v[cluster_type] = cs_c_v
                k_10_topics[cluster_type] = clusters_words

    print("best c_v scores:")
    for m, b_cs in best_c_v.items():
        print(str(m) + ": " + str(b_cs))

    print()
    print("k=10 c_v scores:")
    for m, cs_score in k_10_c_v.items():
        print(str(m) + ": " + str(cs_score))

    # c_v coherence score
    ys = [l for l in y_c_v_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "NMF"], type='c_v')
    fig.savefig("visuals/c_v_alberta_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_u_mass_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="NPMI",
                          color_legends=["K-Means", "Agglomerative", "NMF"], type='c_npmi')
    fig.savefig("visuals/c_npmi_alberta_vs_k.pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_clustering_type.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative", "NMF"], type='dbs')
    fig.savefig("visuals/dbi_alberta_vs_k.pdf", bbox_inches='tight', transparent=True)

    best_c_v_topics_lengths = {"kmeans": None, "agglomerative": None}
    for m, topics in best_c_v_topics.items():
        g, plt = create_circle_tree(topics)
        fig = plt.gcf()
        fig.savefig("visuals/best_" + str(m) + ".pdf", dpi=100, transparent=True)
        nx.write_graphml(g, "visuals/best_" + str(m) + ".graphml")

        # k = 10 model
        g, plt = create_circle_tree(k_10_topics[m])
        nx.write_graphml(g, "visuals/k=10_" + str(m) + ".graphml")

        # add to best_c_v_topics_lengths
        best_c_v_topics_lengths[m] = [len(t) for t in topics]

        # write topics
        write_topics_viz(topics, best_c_v[m], m,
                         "visuals/best_" + str(m) + ".txt")
        write_topics_viz(worst_c_v_topics[m], worst_c_v[m], m,
                         "visuals/worst_" + str(m) + ".txt")
        # write k = 10 model
        write_topics_viz(k_10_topics[m], k_10_c_v[m], m,
                         "visuals/k=10_" + str(m) + ".txt")

    best_topics_lengths = [l for l in best_c_v_topics_lengths.values()]
    _, fig = box_plot(best_topics_lengths, ["K-Means", "Agglomerative", "NMF"], "Clustering Types", "Topic Lengths")
    fig.savefig("visuals/box_plot_alberta.pdf", dpi=100, transparent=True)