from src.model import *
from src.visualizations import *
from src.misc import save_model_scores


def baseline_topic_model(all_data_processed: list, vocab: list, tokenized_docs: list, doc_labels_true: list,
                         test_tokenized_docs: list, x: list = None):

    """

    :param all_data_processed:
    :param vocab:
    :param tokenized_docs:
    :param doc_labels_true:
    :param test_tokenized_docs:
    :param x:
    :return:
    """
    if x is None:
        x = list(range(2, 22, 2))
    else:
        assert isinstance(x, list), "x has to be a list to iterate over"

    true_topic_amount = len(set(doc_labels_true))

    y_topics = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}

    y_c_v_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}
    y_npmi_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}

    test_y_c_v_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}
    test_y_npmi_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}

    doc_topics_pred_model = {'NMF TF': [], 'NMF TF-IDF': [], 'LDA': []}
    for k in x:

        for m in list(y_topics.keys()):

            if m == 'NMF TF':
                topics, doc_topics_pred = nmf_topics(all_data_processed, vocabulary=vocab, n_topics=k, solver='cd',
                                                     beta_loss='frobenius', use_tfidf=False, n_words=50)

            elif m == 'NMF TF-IDF':
                topics, doc_topics_pred = nmf_topics(all_data_processed, vocabulary=vocab, n_topics=k, solver='cd',
                                                     beta_loss='frobenius', use_tfidf=True, n_words=50)

            elif m == 'LDA':
                topics, doc_topics_pred = lda_topics(all_data_processed, n_topics=k,  n_words=50)

            else:
                print(str(m) + "not in :" + str(y_topics.keys()))
                return

            y_topics[m].append(topics)

            # topic evaluation
            # intrinsic scores
            y_c_v_model[m].append(c_v_coherence_score(tokenized_docs, topics, cs_type='c_v'))
            y_npmi_model[m].append(npmi_coherence_score(tokenized_docs, topics, len(topics)))

            # extrinsic scores
            test_y_c_v_model[m].append(c_v_coherence_score(test_tokenized_docs, topics, cs_type='c_v'))
            test_y_npmi_model[m].append(npmi_coherence_score(test_tokenized_docs, topics, len(topics)))

            # save predicted topics assigned for classification evaluation
            doc_topics_pred_model[m].append(doc_topics_pred)

            if k == true_topic_amount:
                label_distribution(doc_labels_true, doc_topics_pred, m)

    save_model_scores(models=list(y_topics.keys()), model_topics=y_topics, model_c_v_scores=y_c_v_model,
                      model_npmi_scores=y_npmi_model, model_c_v_test_scores=test_y_c_v_model,
                      model_npmi_test_scores=test_y_npmi_model, filename_prefix='BL')
    """
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

        vis_classification_score(y_topics[m], m, doc_labels_true, doc_topics_pred_model[m],
                                 filename="visuals/classification_scores_" + str(m) + ".txt",
                                 multiple_true_label_set=True)
                                 
    """