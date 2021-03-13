from src.get_data import *
from src.preprocessing import *
from src.jointly_embedded_space import *
from src.word_space_topicModeling import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from collections import Counter


data, data_labels, test_data, test_data_labels = get_data()

new_data = []
new_data_label = []
for i, d in enumerate(data):
    if len([w for w in d.split() if w.isalpha()]) > 2:
        new_data.append(d)
        new_data_label.append(data_labels[i])
print("removed docs: " + str(len(data) - len(new_data)))

new_test_data = []
new_test_data_label = []
for i, d in enumerate(test_data):

    if len([w for w in d.split() if w.isalpha()]) > 2:
        new_test_data.append(d)
        new_test_data_label.append(test_data_labels[i])
print("removed test docs: " + str(len(test_data) - len(new_test_data)))


# all_data = [d for d in new_data]
# all_data.extend(new_test_data)

# all_data_label = [l for l in new_data_label]
# all_data_label.extend(new_test_data_label)
# assert len(all_data) == len(all_data_label)


# TODO: create a main() function


def number_of_words_per_doc():
    fig, ax = plt.subplots(figsize=(10, 6))

    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.linewidth'] = 2

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')

    ax.yaxis.grid(color='grey', linestyle="--")
    ax.xaxis.grid(alpha=0)

    plt.margins(0)

    all_data_lengths = [len([w for w in doc.split() if w.isalpha()]) for doc in new_data]
    data_lengths_c = [all_data_lengths.count(int(i)) for i in range(int(np.max(all_data_lengths)))]
    plt.bar(range(int(np.max(all_data_lengths))), data_lengths_c, color="black")

    plt.xlim(right=int(np.max(all_data_lengths)))
    plt.xlim(left=0)

    plt.ylim(top=int(np.max(data_lengths_c)))
    plt.ylim(bottom=0)

    ax.set_xlabel("Number of Words", fontsize="medium")
    ax.set_ylabel("Number of Segments", fontsize="medium")

    plt.show()
    fig.savefig("visuals/segment_word_distribution.pdf", bbox_inches='tight', transparent=True)


def vis_most_common_words(data: list, raw_data: False, preprocessed: False):
    if raw_data:
        data = [doc.split() for doc in data]
        y_max = 25000
        filename = "most_common_words"
    else:
        if preprocessed:

            y_max = 1000
        else:

            y_max = 4000
        filename = "processed_most_common_words"

    fig, ax = plt.subplots(figsize=(10, 6))

    mpl.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 2

    ax.tick_params(axis='both', labelsize=12)

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')

    ax.yaxis.grid(color='grey', linestyle="--", alpha=0.5)
    ax.xaxis.grid(alpha=0)
    plt.margins(0)

    data_words = []
    for doc in data:
        data_words.extend([w.lower() for w in doc if w.isalpha()])

    data_words_c = Counter(data_words)

    most_common_words = [w for w, c in data_words_c.most_common(30)]
    most_common_words_c = [c for w, c in data_words_c.most_common(30)]

    plt.bar(most_common_words, most_common_words_c, color='black', width=0.5)

    plt.ylim(top=y_max)
    plt.ylim(bottom=0)

    ax.set_xlabel("Top 30 Words", fontsize="medium")
    ax.set_ylabel("Number of Occurrences", fontsize="medium")

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

    fig.savefig("visuals/" + str(filename) + ".pdf", bbox_inches='tight', transparent=True)




if __name__ == "__main__":

    do_lemmatizing = False
    do_stop_word_removal = False

    data_processed, data_labels, vocab, tokenized_docs = preprocessing(
        new_data, new_data_label, do_lemmatizing=do_lemmatizing, do_stop_word_removal=do_stop_word_removal)

    test_data_processed, test_data_labels, test_vocab, test_tokenized_docs = preprocessing(
        new_test_data, new_test_data_label, do_lemmatizing=do_lemmatizing, do_stop_word_removal=do_stop_word_removal)

    #####
    # document space
    #####
    # baseline_topic_model(data_processed, vocab, tokenized_docs, data_labels, test_tokenized_docs)
    # not used: doc_clustering(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="w2v_avg")

    #####
    # word space
    #####
    # re_ranking_topic_model(data_processed, vocab, tokenized_docs, test_tokenized_docs)
    # topic_vector_model(data_processed, vocab, tokenized_docs, test_tokenized_docs)
    # k_components_model(data_processed, vocab, tokenized_docs, test_tokenized_docs)
    # bert_visualization(data_processed, vocab, test_tokenized_docs)

    ####
    # word + doc space
    ####
    w_d_clustering(data_processed, vocab, tokenized_docs, data_labels, test_tokenized_docs,
                   doc_embedding_type="doc2vec", true_topic_amount=8)
    # test_clustering(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="w2v_avg")

    # doc_clustering(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="w2v_avg")

    # bert_visualization(all_data_processed, vocab, tokenized_docs)
    # w_d_get_graph_components(all_data_processed, vocab, tokenized_docs, all_data_labels, doc_embedding_type="w2v_avg")


    ####
    # Misc
    ####
    # number_of_words_per_doc()
    # vis_most_common_words(data_processed, raw_data=False, preprocessed=True)




