from src.visualizations import *
from src.vectorization import *
from src.clustering import *


def get_w2v_vis_sign_words(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):
    clustering_weight_type = 'tf'
    ranking_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    # _, _, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")
    # w2v_model
    # min_count: 50
    # window: 3
    # negative: 60
    # ns_exponent: 0.75
    # orig = w2v_model.min_count
    # print(orig)
    # w2v_params = {'min_c': k, 'win': w2v_model.window, 'negative': w2v_model.negative, 
    # 'ns_exponent': w2v_model.ns_exponent, 'seed': 42}

    w2v_params = {"min_c": 50, "win": 15, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42}

    y_c_v_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    y_dbs_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}

    y_topics = {'kmeans': [], 'agglomerative': [], 'hdbscan': []}

    words, word_embeddings, _ = get_word_vectors(all_data_processed, vocab, params=w2v_params)

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "hdbscan"]:
            
            if cluster_type == "hdbscan":

                labels, probabilities = hdbscan_clustering(word_embeddings, min_cluster_size=6, do_dim_reduction=False)

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
                assert cluster_type in ["kmeans", "agglomerative"]
                if cluster_type == "kmeans":
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
            cs_npmi = average_npmi_topics(tokenized_docs, clusters_words, len(clusters_words))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_dbs_model[cluster_type].append(dbs)

            y_topics[cluster_type].append(clusters_words)

    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/ws_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/ws_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        assert len(y_c_v_model[m]) == len(y_npmi_model[m])
        assert len(y_npmi_model[m]) == len(y_dbs_model[m])
        assert len(y_dbs_model[m]) == len(y_topics[m])
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], "visuals/ws_clusters_eval_" + str(m) + ".txt",
                         dbs_scores=y_dbs_model[m])


def get_w2v_vis_topic_vec(all_data_processed: list, vocab: list, tokenized_docs: list, x: list = None):
    clustering_weight_type = 'tf'

    if x is None:
        x = list(range(2, 22, 2))

    else:
        assert isinstance(x, list)

    # _, _, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")
    # w2v_model
    # min_count: 50
    # window: 3
    # negative: 60
    # ns_exponent: 0.75
    # orig = w2v_model.min_count
    # print(orig)
    # w2v_params = {'min_c': k, 'win': w2v_model.window, 'negative': w2v_model.negative,
    # 'ns_exponent': w2v_model.ns_exponent, 'seed': 42}

    w2v_params = {"min_c": 50, "win": 15, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42}

    y_c_v_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    y_dbs_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}
    y_npmi_model = {"kmeans": [], "agglomerative": [], "hdbscan": []}

    y_topics = {'kmeans': [], 'agglomerative': [], 'hdbscan': []}

    words, word_embeddings, _ = get_word_vectors(all_data_processed, vocab, params=w2v_params)

    for k in x:

        for cluster_type in ["kmeans", "agglomerative", "hdbscan"]:

            if cluster_type == "hdbscan":
                labels, probabilities = hdbscan_clustering(word_embeddings, min_cluster_size=6, do_dim_reduction=False)

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
                assert cluster_type in ["kmeans", "agglomerative"]
                if cluster_type == "kmeans":
                    clustering_params = {'n_clusters': k, 'random_state': 42, }
                else:
                    clustering_params = {'n_clusters': k}

                clusters_words, clusters_words_embeddings = word_clusters(
                    all_data_processed, words, word_embeddings, vocab, clustering_type=cluster_type,
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

            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, topic_vector_cluster_words, cs_type='c_v')))
            dbs = float("{:.2f}".format(davies_bouldin_index(topic_vector_cluster_words_embeddings)))
            cs_npmi = average_npmi_topics(tokenized_docs, topic_vector_cluster_words, len(topic_vector_cluster_words))

            y_c_v_model[cluster_type].append(cs_c_v)
            y_npmi_model[cluster_type].append(cs_npmi)
            y_dbs_model[cluster_type].append(dbs)

            y_topics[cluster_type].append(topic_vector_cluster_words)

    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_v')
    fig.savefig("visuals/ws_c_v_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["K-Means", "Agglomerative", "HDBSCAN"], type='c_npmi')
    fig.savefig("visuals/ws_c_npmi_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], "visuals/ws_clusters_eval_" + str(m) + ".txt",
                         dbs_scores=y_dbs_model[m])


"""
# _, _, w2v_model = get_word_vectors(all_data_processed, vocab, "data/w2v_node2vec")
    # print("min_count: " + str(w2v_model.min_count))
    # print("window: " + str(w2v_model.window))
    # print("negative: " + str(w2v_model.negative))
    # print("ns_exponent: " + str(w2v_model.ns_exponent))

    # w2v_params = {'min_c': w2v_model.min_count, 'win': w2v_model.window, 'negative': w2v_model.negative,
    # 'seed': 42, 'sample': w2v_model.sample}

    # top2vec
    # w2v_params = {"min_c": 50, "win": 15, "negative": 0, "sample": 1e-5, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42}

    # mix of params
    # w2v_params = {"min_c": w2v_model.min_count, "win": w2v_model.window, "sample": w2v_model.sample, 
    # "negative": w2v_model.negative, "hs": 1, "epochs": 400, "sg": 1, 'seed': 42, 'cbow_mean': 1}

    # words, word_embeddings, _ = get_word_vectors(all_data_processed, vocab, params=w2v_params)






# c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label=x_label, y_label="Coherence Score (c_v)",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='c_v')
    fig.savefig("visuals/c_v_w2v_vs_" + str(file_save_under) + ".pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label=x_label, y_label="NPMI",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='c_npmi')
    fig.savefig("visuals/npmi_w2v_vs_" + str(file_save_under) + ".pdf", bbox_inches='tight', transparent=True)

    # dbs score
    ys = [l for l in y_dbs_model.values()]
    _, fig = scatter_plot(x, ys, x_label=x_label, y_label="Davies–Bouldin index",
                          color_legends=["K-Means", "Agglomerative", "Spectral"], type='dbs')
    fig.savefig("visuals/dbi_w2v_vs_" + str(file_save_under) + ".pdf", bbox_inches='tight', transparent=True)

    best_c_v_topics_lengths = {"kmeans": None, "agglomerative": None, "spectral": None}
    for m, topics in best_c_v_topics.items():
        g, plt = create_circle_tree(topics)
        fig = plt.gcf()
        fig.savefig("visuals/best_" + str(m) + "_" + str(file_save_under) + ".pdf", dpi=100, transparent=True)
        nx.write_graphml(g, "visuals/best_" + str(m) + "_" + str(file_save_under) + ".graphml")

        # add to best_c_v_topics_lengths
        best_c_v_topics_lengths[m] = [len(t) for t in topics]

        # write topics
        write_topics_ablation(topics, best_var[m], "visuals/best_" + str(m) + "_" + str(file_save_under) + ".txt")
        write_topics_ablation(topics, worst_var[m],
                              "visuals/worst_" + str(m) + "_" + str(file_save_under) + ".txt")

    best_topics_lengths = [l for l in best_c_v_topics_lengths.values()]
    _, fig = box_plot(best_topics_lengths, ["K-Means", "Agglomerative", "Spectral"], "Clustering Methods",
                      "Topic Lengths")
    fig.savefig("visuals/box_plot_w2v_" + str(file_save_under) + ".pdf", dpi=100, transparent=True)


"""