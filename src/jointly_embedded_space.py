from src.visualizations import *
from src.vectorization import *
from src.clustering import *
from src.graphs import *


def w_d_clustering(all_data_processed: list, vocab: list, tokenized_docs: list, doc_labels_true: list,
                   test_tokenized_docs: list, doc_embedding_type="w2v_avg", x: list = None, true_topic_amount=10):

    # main extrinsic evaluation metric: ARI
    # https://stats.stackexchange.com/questions/381223/evaluation-of-clustering-method

    # true_topic_amount = len(set(doc_labels_true))

    if x is None:
        x = list(range(2, 22, 2))
        assert true_topic_amount in x

    else:
        assert isinstance(x, list)

    if doc_embedding_type in ["w2v_avg", "w2v_sum"]:
        params = {"min_c": 10, "win": 7, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42,
                  'ns_exponent': 0.75}
        min_cluster_size = 7  # fully preprocessed: 9, just nouns: 7

    else:
        assert doc_embedding_type == "doc2vec"
        params = {"min_c": 10, "win": 5, "negative": 30, "sample": 1e-5, "hs": 0, "epochs": 400, 'seed': 42,
                  'ns_exponent': 0.75, "dm": 0, "dbow_words": 1}
        min_cluster_size = 8

    doc_data, short_true_labels, doc_embeddings, vocab_words, vocab_embeddings = get_doc_embeddings(
        all_data_processed, doc_labels_true, vocab, doc_embedding_type, params)

    y_topics = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}
    y_c_v_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}

    test_y_c_v_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}
    test_y_npmi_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}

    doc_topics_pred_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}
    doc_topics_true_model = {"kmeans": [], "agglomerative": [], "HDBSCAN": []}

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

            cs_c_v = float("{:.2f}".format(c_v_coherence_score(all_data_processed, topics_words, cs_type='c_v')))
            print(cluster_type)
            print("c_v: " + str(cs_c_v))
            cs_npmi = float("{:.2f}".format(npmi_coherence_score(all_data_processed, topics_words, len(topics_words))))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_topics[cluster_type].append(topics_words)

            # extrinsic
            test_y_c_v_model[cluster_type].append(float("{:.2f}".format(c_v_coherence_score(test_tokenized_docs,
                                                                                            topics_words, cs_type='c_v'))))
            test_y_npmi_model[cluster_type].append(npmi_coherence_score(test_tokenized_docs, topics_words,
                                                                        len(topics_words)))

            if k == true_topic_amount:
                label_distribution(true_labels, labels_predict, cluster_type)

            # save predicted topics assigned for classification evaluation
            doc_topics_pred_model[cluster_type].append(labels_predict)
            doc_topics_true_model[cluster_type].append(true_labels)

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
    plt.close(fig)

    # extrinsic
    # c_v coherence score
    ys = [l for l in test_y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/extrinsic_w_d_space_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig)

    # NMPI coherence score
    ys = [l for l in test_y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/extrinsic_w_d_space_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], test_y_c_v_model[m], test_y_npmi_model[m],
                         "visuals/w_d_space_clusters_eval_" + str(m) + ".txt")

        vis_classification_score(y_topics[m], m, doc_topics_true_model[m], doc_topics_pred_model[m],
                                 filename="visuals/classification_scores_" + str(m) + ".txt",
                                 multiple_true_label_set=True)
