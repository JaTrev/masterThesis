from src.model import *
from src.evaluation import *
from src.visualizations import *
from src.vectorization import *
from src.clustering import *


def get_baseline(all_data_processed: list, vocab: list, tokenized_docs: list, doc_labels_true: list,
                 test_tokenized_docs: list, x: list = None):
    if x is None:
        x = list(range(2, 22, 2))
    else:
        assert isinstance(x, list), "x has to be a list to iterate over"

    true_topic_amount = len(set(doc_labels_true))

    y_topics = {'nmf_tf': [], 'nmf_tf_idf': [], 'lda': []}

    y_c_v_model = {"nmf_tf": [], "nmf_tf_idf": [], "lda": []}
    y_npmi_model = {"nmf_tf": [], "nmf_tf_idf": [], "lda": []}

    test_y_c_v_model = {"nmf_tf": [], "nmf_tf_idf": [], "lda": []}
    test_y_npmi_model = {"nmf_tf": [], "nmf_tf_idf": [], "lda": []}

    for k in x:

        for m in list(y_topics.keys()):

            if m == 'nmf_tf':
                topics, doc_topics_pred = nmf_topics(all_data_processed, vocabulary=vocab, n_topics=k, solver='cd',
                                                     beta_loss='frobenius', use_tfidf=False, n_words=50)

            elif m == 'nmf_tf_idf':
                topics, doc_topics_pred = nmf_topics(all_data_processed, vocabulary=vocab, n_topics=k, solver='cd',
                                                     beta_loss='frobenius', use_tfidf=True, n_words=50)

            elif m == 'lda':
                topics, doc_topics_pred = lda_topics(all_data_processed, n_topics=k,  n_words=50)

            else:
                print(str(m) + "not in :" + str(y_topics.keys()))
                return

            cs_c_v = float("{:.2f}".format(coherence_score(tokenized_docs, topics, cs_type='c_v')))
            cs_npmi = average_npmi_topics(tokenized_docs, topics, len(topics))

            y_c_v_model[m].append(cs_c_v)
            y_npmi_model[m].append(cs_npmi)

            test_y_c_v_model[m].append(float("{:.2f}".format(coherence_score(test_tokenized_docs,
                                                                             topics, cs_type='c_v'))))
            test_y_npmi_model[m].append(average_npmi_topics(test_tokenized_docs, topics, len(topics)))

            y_topics[m].append(topics)

            if k == true_topic_amount:
                vis_classification_score(m, doc_labels_true, doc_topics_pred, topics, "visuals/classification_scores_"
                                         + str(m) + ".txt")

                label_distribution(doc_labels_true, doc_topics_pred, m)

    # intrinsic
    # c_v coherence score
    ys = [l for l in y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["NMF TF", "NMF TF-IDF", "LDA"], type='c_v')
    fig.savefig("visuals/c_v_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)
    # close fig
    plt.close(fig)

    # NMPI coherence score
    ys = [l for l in y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["NMF TF", "NMF TF-IDF", "LDA"], type='c_npmi')
    fig.savefig("visuals/c_npmi_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)
    # close fig
    plt.close(fig)

    # extrinsic
    # c_v coherence score
    ys = [l for l in test_y_c_v_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (c_v)",
                          color_legends=["NMF TF", "NMF TF-IDF", "LDA"], type='c_v')
    fig.savefig("visuals/extrinsic_c_v_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)
    # close fig
    plt.close(fig)

    # NMPI coherence score
    ys = [l for l in test_y_npmi_model.values()]
    _, fig = scatter_plot(x, ys, x_label="Number of Topics", y_label="Coherence Score (NMPI)",
                          color_legends=["NMF TF", "NMF TF-IDF", "LDA"], type='c_npmi')
    fig.savefig("visuals/extrinsic_c_npmi_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)
    # close fig
    plt.close(fig)

    for m in list(y_topics.keys()):
        vis_topics_score(y_topics[m], y_c_v_model[m], y_npmi_model[m], test_y_c_v_model[m], test_y_npmi_model[m],
                         "visuals/clusters_eval_" + str(m) + ".txt")

