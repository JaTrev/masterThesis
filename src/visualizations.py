import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.evaluation import *
import os
import numpy as np
from collections import Counter


def vis_classification_score(topics_list: list, model_type: str, doc_labels_true: list, doc_topics_pred_list: list,
                             filename, n_words=10, multiple_true_label_set=False):
    """

    :param topics_list:
    :param model_type:
    :param doc_labels_true:
    :param doc_topics_pred_list:
    :param filename:
    :param n_words:
    :param multiple_true_label_set:
    :return:
    """
    # filename = "visuals/classification_scores.txt"

    if any([len(t) > n_words for topics in topics_list for t in topics]):
        new_topics_list = [[t[:10] for t in topics] for topics in topics_list]
        topics_list = new_topics_list

    with open(filename, "w") as myFile:
        myFile.write('Model:  ' + str(model_type) + '\n')

        for i, topics in enumerate(topics_list):

            for i_t, t in enumerate(topics):
                myFile.write('Topic ' + str(i_t + 1) + '\n')

                for w in t:
                    myFile.write(str(w) + ' ')

                myFile.write('\n')

            myFile.write('\n')

            if multiple_true_label_set:
                true_labels = doc_labels_true[i]

            else:
                true_labels = doc_labels_true
            myFile.write("ari score: " + ": " + str(ari_score(true_labels, doc_topics_pred_list[i])) + '\n')
            myFile.write("ami score: " + ": " + str(ami_score(true_labels, doc_topics_pred_list[i])) + '\n')
            myFile.write("nmi score: " + ": " + str(nmi_score(true_labels, doc_topics_pred_list[i])) + '\n')

            myFile.write('\n\n\n')

    myFile.close()


def label_distribution(doc_labels_true: list, doc_topics_pred: list, model_name: str):
    """

    :param doc_labels_true:
    :param doc_topics_pred:
    :param model_name:
    :return:
    """

    assert len(doc_labels_true) == len(doc_topics_pred), "labels must have same length"

    labels_true = np.array(doc_labels_true)
    labels_predicted = np.array(doc_topics_pred)

    parent_dir = "visuals/" + model_name + "_dir"
    os.mkdir(parent_dir)

    topics = set(doc_topics_pred)
    assert -1 not in topics

    for t in range(len(topics)):
        predicted_indices = np.argwhere(labels_predicted == t)

        t_true = labels_true[predicted_indices].flatten()

        fig, ax = vis_prep()

        ax.set_xlabel("T$_{" + str(t+1) + "}$'s " + "True Topic Distribution", fontsize='medium', labelpad=4)
        ax.set_ylabel("Number of Segments", fontsize='medium', labelpad=4)
        ax.tick_params(axis='both', labelsize='small')
        plt.setp(ax.spines.values(), linewidth=2)
        plt.grid(color='grey', axis='y', linestyle='--', linewidth=0.7)

        labels, values = zip(*Counter(t_true).items())

        values = list(values)
        max_values = int(max(values)/20)*20 + 40

        bins = np.arange(len(set(t_true))+1) - 0.5

        colors = ["#d14035", "#eb8a3c", "#ebb481", "#775845", "#31464f", "#86aa40", "#33655b", "#7ca2a1", "#B9EDF8",
                  "#39BAE8"]

        t_true_list = [[l for l in t_true if i == l] for i in range(10)]

        # plt.hist(t_true_list, ec="white") #      bins,    plt.hist(t_true_list, bins,   ec="white", rwidth=)
        plt.bar(range(len(t_true_list)), height=[len(l) for l in t_true_list], width=0.8, color=colors[:len(t_true_list)])

        plt.xticks(list(range(10)), list(range(1, 11)))
        ax.yaxis.set_ticks(list(range(0, max_values, 20)))

        fig.savefig(parent_dir + "/topic" + str(t+1) + ".pdf", bbox_inches='tight', transparent=True)

        # close fig
        plt.close(fig)


def vis_topics_score(topics_list: list, c_v_scores: list, nmpi_scores: list, test_c_v_scores: list,
                     test_nmpi_scores: list, filename: str, dbs_scores: list = None, n_words: int = 10):
    """

    :param topics_list:
    :param c_v_scores:
    :param nmpi_scores:
    :param test_c_v_scores:
    :param test_nmpi_scores:
    :param filename:
    :param dbs_scores:
    :param n_words:
    :return:
    """
    assert len(topics_list) == len(c_v_scores)
    assert len(c_v_scores) == len(nmpi_scores)

    if dbs_scores is not None:
        assert len(nmpi_scores) == len(dbs_scores)

    if any([len(t) > n_words for topics in topics_list for t in topics]):
        new_topics_list = [[t[:10] for t in topics] for topics in topics_list]
        topics_list = new_topics_list

    with open(filename, "w") as myFile:

        for i, topics in enumerate(topics_list):

            for i_t, t in enumerate(topics):
                myFile.write('Topic ' + str(i_t + 1) + '\n')

                for w in t:
                    myFile.write(str(w) + ' ')

                myFile.write('\n')

            myFile.write("intrinsic evaluation" + '\n')
            myFile.write("c_v score: " + str(c_v_scores[i]) + '\n')
            myFile.write("nmpi score: " + str(nmpi_scores[i]) + '\n')

            myFile.write("extrinsic evaluation" + '\n')
            myFile.write("c_v score: " + str(test_c_v_scores[i]) + '\n')
            myFile.write("nmpi score: " + str(test_nmpi_scores[i]) + '\n')

            if dbs_scores is not None:
                myFile.write("dbs score: " + ": " + str(dbs_scores[i]) + '\n')

            myFile.write('\n\n\n')

    myFile.close()


def vis_prep():
    """

    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='both', labelsize=12)

    # mpl.rcParams['font.family'] = 'Avenir'
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
    # plt.margins(0)

    return fig, ax


def scatter_plot(x: list, ys: list, x_label: str, y_label: str, color_legends: list, type: str) -> plt:
    """

    :param x:
    :param ys:
    :param x_label:
    :param y_label:
    :param color_legends:
    :param type:
    :return:
    """
    fig, ax = vis_prep()

    assert type in ["c_v", "u_mass", "c_npmi", "dbs", "ari", "ami", "acc"], "define the type of scatter plot"
    assert isinstance(ys[0], list), "ys needs to be a list of list(s)"
    assert len(ys) == len(color_legends), "need a color legend for each y list (ys)"

    mapper = mpl.cm.get_cmap('Pastel2')
    ys_color = [mapper(i_y) for i_y,_ in enumerate(ys)]

    error_value = -1000
    for i_y, y in enumerate(ys):
        new_y = [value if value != error_value else np.nan for value in y]
        plt.plot(x, new_y, 'o-', c=ys_color[i_y], markersize=17, linewidth=3, label=color_legends[i_y])

    if type == "u_mass":
        y_ticks = [x for x in range(-6, 8, 2)]

    elif type == "c_npmi":
        y_ticks = [x/10 for x in range(-1, 6, 1)]

    elif type == "c_v":
        y_ticks = [x / 10 for x in range(0, 11, 1)]

    elif type == "dbs":
        all_y = []
        for y in ys:
            all_y.extend(y)
        y_ticks = [x/10 for x in range(00, 40, 5)]
    else:
        assert type in ["ari", "ami", "acc"]
        y_ticks = [x / 10 for x in range(0, 7, 1)]

    ax.set_xlabel(x_label, fontsize='medium', labelpad=4)
    ax.set_ylabel(y_label, fontsize='medium', labelpad=4)

    ax.yaxis.set_ticks(y_ticks)
    ax.xaxis.set_ticks(x)

    ax.tick_params(axis='both', labelsize='small')

    plt.legend(fontsize='x-small')
    plt.setp(ax.spines.values(), linewidth=2)
    plt.grid(color='grey', axis='y', linestyle='--', linewidth=0.7)
    return plt, fig


def tsne_plot(clusters: list, cluster_embeddings: list,
              perplexity: int = 40, n_iter: int = 500, random_state: int = 42):
    """
    Create a TSNE 2-D plot.
    :param clusters: list of words for each cluster
    :param cluster_embeddings: list of embeddings for each cluster
    :param perplexity: perplexity for TSNE
    :param n_iter: number of iteration for TSNE
    :param random_state: random state for TSNE
    :return: scatter plot of TSNE
    """

    words = []
    word_embeddings = []
    for i_c in range(len(clusters)):
        words.extend(clusters[i_c])
        word_embeddings.extend(cluster_embeddings[i_c])

    assert len(words) == len(word_embeddings), "make sure the number of words and the number of word embeddings " \
                                               "is the same"

    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iter, random_state=random_state)
    new_values = tsne_model.fit_transform(word_embeddings)

    x_coord, y_coord = [], []
    for value in new_values:
        x_coord.append(value[0])
        y_coord.append(value[1])

    norm = mpl.colors.Normalize(vmin=0, vmax=len(clusters) + 1, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gist_ncar)

    node_colors = []
    for i_c, c in enumerate(clusters):
        node_colors.extend([mapper.to_rgba(i_c) for _ in c])

    for i in range(len(node_colors)):
        plt.scatter(x_coord[i], y_coord[i], c=node_colors[i])
        plt.annotate(words[i],
                     xy=(x_coord[i], y_coord[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    return plt
