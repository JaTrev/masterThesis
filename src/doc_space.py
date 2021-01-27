from src.model import *
from src.evaluation import *
from src.visualizations import *
from src.vectorization import *
from src.clustering import *


def get_baseline(all_data_processed: list, vocab: list, tokenized_docs: list, doc_labels_true: list, x: list = None):
    if x is None:
        x = list(range(2, 22, 2))
    else:
        assert isinstance(x, list), "x has to be a list to iterate over"

    true_topic_amount = len(set(doc_labels_true))

    y_topics = {'nmf_tf': [], 'nmf_tf_idf': [], 'lda': []}

    y_c_v_model = {"nmf_tf": [], "nmf_tf_idf": [], "lda": []}
    y_npmi_model = {"nmf_tf": [], "nmf_tf_idf": [], "lda": []}

    for k in x:

        for m in list(y_topics.keys()):

            if m == 'nmf_tf':
                topics, doc_topics_pred = nmf_topics(all_data_processed, vocabulary=vocab, n_topics=k, solver='cd',
                                                     beta_loss='frobenius', use_tfidf=False)

            elif m == 'nmf_tf_idf':
                topics, doc_topics_pred = nmf_topics(all_data_processed, vocabulary=vocab, n_topics=k, solver='cd',
                                                     beta_loss='frobenius', use_tfidf=True)

            elif m == 'lda':
                topics, doc_topics_pred = lda_topics(all_data_processed, n_topics=k)

            else:
                print(str(m) + "not in :" + str(y_topics.keys()))
                return

            # todo: implement dbs for baseline
            # dbs = float("{:.2f}".format(davies_bouldin_index(clusters_words_embeddings)))
            # y_dbs_clustering_type[m].append(dbs)

            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, topics, cs_type='c_v')))
            cs_npmi = average_npmi_topics(tokenized_docs, topics, len(topics))

            y_c_v_model[m].append(cs_c_v)
            y_npmi_model[m].append(cs_npmi)

            y_topics[m].append(topics)

            if k == true_topic_amount:
                vis_classification_score(m, doc_labels_true, doc_topics_pred, topics,
                                     "visuals/classification_scores_" + str(m) + ".txt")

    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["NMF TF", "NMF TF-IDF", "LDA"], type='c_v')
    fig.savefig("visuals/c_v_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)

    # u_mass coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["NMF TF", "NMF TF-IDF", "LDA"], type='c_npmi')
    fig.savefig("visuals/c_npmi_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], "visuals/clusters_eval_" + str(m) + ".txt")


def doc_clustering(all_data_processed: list, vocab: list, tokenized_docs: list, doc_labels_true: list,
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

    doc_data, true_labels, doc_embeddings, vocab_words, vocab_embeddings = get_doc_embeddings(
        all_data_processed, doc_labels_true, vocab, doc_embedding_type, d2v_params)

    y_topics = {"kmeans": [], "agglomerative": []}
    y_c_v_model = {"kmeans": [], "agglomerative": []}
    y_npmi_model = {"kmeans": [], "agglomerative": []}

    for k in x:

        for cluster_type in ["kmeans", "agglomerative"]:

            if cluster_type in ["kmeans", "nmf"]:
                clustering_params = {'n_clusters': k, 'random_state': 42, }
            else:
                clustering_params = {'n_clusters': k}

            # clustering document via document embeddings
            clusters_docs, clusters_docs_embeddings, labels_predict, doc_data = document_clustering(doc_data,
                                                                                                    doc_embeddings,
                                                                                                    vocab, cluster_type,
                                                                                                    params=
                                                                                                    clustering_params)
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
                          color_legends=["K-Means", "Agglomerative"], type='c_v')
    fig.savefig("visuals/c_v_doc_vs_k.pdf", bbox_inches='tight', transparent=True)

    # npmi coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (npmi)",
                          color_legends=["K-Means", "Agglomerative"], type='c_npmi')
    fig.savefig("visuals/npmi_doc_vs_k.pdf", bbox_inches='tight', transparent=True)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], "visuals/d2v_clusters_eval_" + str(m) + ".txt")
