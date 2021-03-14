from src.get_data import *
from src.preprocessing import *
from src.doc_space_topicModeling import *
from src.jointly_embedded_space import *
from src.word_space_topicModeling import *
import argparse


def main_fct(preprocessed_data_set: str, topic_model_type: str, misc_prints: str,
             bert_embedding_type: str = "normal_ll"):
    """
    main function performing all topic models

    :param preprocessed_data_set: name of preprocessed data set
    :param topic_model_type: name of topic model
    :param misc_prints: misc. print functions
    :param bert_embedding_type: BERT embedding type
    :return:
    """

    assert preprocessed_data_set in ["JN", "FP"]
    assert (topic_model_type in ["Baseline", "RRW", "TVS", "k-components", "BERT", "avg_w2v", "doc2vec"]
            or misc_prints in ["segment_size", "common_words"])

    if preprocessed_data_set == "JN":
        do_lemmatizing = False
        do_stop_word_removal = False
    else:
        do_lemmatizing = True
        do_stop_word_removal = True

    data_processed, data_processed_labels, vocab, tokenized_docs = preprocessing(
        new_data, new_data_label, do_lemmatizing=do_lemmatizing, do_stop_word_removal=do_stop_word_removal)

    test_data_processed, test_data_processed_labels, test_vocab, test_tokenized_docs = preprocessing(
        new_test_data, new_test_data_label, do_lemmatizing=do_lemmatizing, do_stop_word_removal=do_stop_word_removal)

    if topic_model_type == "Baseline":
        baseline_topic_model(data_processed, vocab, tokenized_docs, data_processed_labels, test_tokenized_docs)

    elif topic_model_type == "RRW":
        word2vec_topic_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                             data_set_name=preprocessed_data_set, topic_vector_flag=False)

    elif topic_model_type == "TVS":
        word2vec_topic_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                             data_set_name=preprocessed_data_set, topic_vector_flag=True)

    elif topic_model_type == "k-components":
        k_components_model(data_processed, vocab, tokenized_docs, test_tokenized_docs,
                           data_set_name=preprocessed_data_set)

    elif topic_model_type == "BERT":
        bert_topic_model(bert_embedding_type, data_processed, vocab, test_tokenized_docs)

    elif topic_model_type == "avg_w2v":
        w_d_clustering(data_processed, preprocessed_data_set, vocab, data_processed_labels,
                       test_tokenized_docs, segment_embedding_type="w2v_avg", true_topic_amount=10)

    elif topic_model_type == "doc2vec":
        w_d_clustering(data_processed, preprocessed_data_set, vocab, data_processed_labels,
                       test_tokenized_docs, segment_embedding_type="doc2vec", true_topic_amount=10)

    else:
        if misc_prints == "segment_size":
            temp_segments = [seg for seg in new_data]
            temp_segments.extend(new_test_data)
            number_of_words_per_doc(temp_segments)

        else:
            assert misc_prints == "common_words", "must define a valid topic_model_type or a misc_prints"
            vis_most_common_words(data_processed, raw_data=False, preprocessed=True)


if __name__ == "__main__":

    data, data_labels, test_data, test_data_labels = get_data()

    new_data = []
    new_data_label = []
    for i, d in enumerate(data):
        if len([w for w in d.split() if w.isalpha()]) > 2:
            new_data.append(d)
            new_data_label.append(data_labels[i])
    print("removed training segments: " + str(len(data) - len(new_data)))

    new_test_data = []
    new_test_data_label = []
    for i, d in enumerate(test_data):
        if len([w for w in d.split() if w.isalpha()]) > 2:
            new_test_data.append(d)
            new_test_data_label.append(test_data_labels[i])
    print("removed test segments: " + str(len(test_data) - len(new_test_data)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--pds', dest='data_set', type=str, required=True,
                        help="state the preprocessed data set should be used: Just Nouns (JN), Fully Preprocessed (FP)")

    parser.add_argument('--tm', dest='topic_model', type=str, required=True,
                        help="state what topic model that should be used")

    parser.add_argument('--mp', dest='misc_prints', type=str, required=False, default=None,
                        help="define the miscellaneous print that should be performed")

    parser.add_argument('--bert', dest='bert_type', type=str, required=False, default=None,
                        help="define the BERT embedding type")

    args = parser.parse_args()

    assert args.data_set in ["JN", "FP"], "need to select a proper preprocessing schema [JN, FP]"
    assert args.topic_model in ['Baseline', 'RRW', 'TVS', 'k-components', 'BERT', 'avg_w2v', 'doc2vec', 'null'], (
        "select one of the topic models: ['Baseline', 'RRW', 'TVS', 'k-components', 'BERT', 'avg_w2v', 'doc2vec', "
        "'null']")

    if args.topic_model == "null":
        assert args.misc_prints in ['segment_size', 'common_words'], ("select one of the misc_prints:  "
                                                                      "['segment_size', 'common_words']")

    main_fct(preprocessed_data_set=args.data_set, topic_model_type=args.topic_model, misc_prints=args.misc_prints,
             bert_embedding_type=args.bert_type)
