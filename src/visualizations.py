import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from math import sqrt
from sklearn.manifold import TSNE


def vis_prep():
    fig, ax = plt.subplots(figsize=(10, 8))
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


def sigma(coords, x, y, r):
    """Computes Sigma for circle fit."""
    dx, dy, sum_ = 0., 0., 0.

    for i in range(len(coords)):
        dx = coords[i][1] - x
        dy = coords[i][0] - y
        sum_ += (sqrt(dx*dx+dy*dy) - r)**2
    return sqrt(sum_/len(coords))


def hyper_fit(coords: list, IterMax= 99, verbose=False):
    """
    Fits coords to circle using hyperfit algorithm.
    Inputs:
        - coords, list or numpy array with len>2 of the form:
        [
    [x_coord, y_coord],
    ...,
    [x_coord, y_coord]
    ]
        or numpy array of shape (n, 2)
    Outputs:
        - xc : x-coordinate of solution center (float)
        - yc : y-coordinate of solution center (float)
        - R : Radius of solution (float)
        - residu : s, sigma - variance of data wrt solution (float)

    Code from:
    Circle-Fit
    https://github.com/AlliedToasters/circle-fit
    by Michael Klear / AlliedToasters and Marian KLeineberg
    December, 2020
    """
    X, X = None, None
    if isinstance(coords, np.ndarray):
        X = coords[:, 0]
        Y = coords[:, 1]
    elif isinstance(coords, list):
        X = np.array([x[0] for x in coords])
        Y = np.array([x[1] for x in coords])
    else:
        raise Exception("Parameter 'coords' is an unsupported type: " + str(type(coords)))

    n = X.shape[0]

    Xi = X - X.mean()
    Yi = Y - Y.mean()
    Zi = Xi*Xi + Yi*Yi

    # compute moments
    Mxy = (Xi*Yi).sum()/n
    Mxx = (Xi*Xi).sum()/n
    Myy = (Yi*Yi).sum()/n
    Mxz = (Xi*Zi).sum()/n
    Myz = (Yi*Zi).sum()/n
    Mzz = (Zi*Zi).sum()/n

    # computing the coefficients of characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx*Myy - Mxy*Mxy
    Var_z = Mzz - Mz*Mz

    A2 = 4*Cov_xy - 3*Mz*Mz - Mzz
    A1 = Var_z*Mz + 4.*Cov_xy*Mz - Mxz*Mxz - Myz*Myz
    A0 = Mxz*(Mxz*Myy - Myz*Mxy) + Myz*(Myz*Mxx - Mxz*Mxy) - Var_z*Cov_xy
    A22 = A2 + A2

    # finding the root of the characteristic polynomial
    y = A0
    x = 0.
    for i in range(IterMax):
        Dy = A1 + x*(A22 + 16.*x*x)
        xnew = x - y/Dy
        if xnew == x or not np.isfinite(xnew):
            break
        ynew = A0 + xnew*(A1 + xnew*(A2 + 4.*xnew*xnew))
        if abs(ynew)>=abs(y):
            break
        x, y = xnew, ynew

    det = x*x - x*Mz + Cov_xy
    Xcenter = (Mxz*(Myy - x) - Myz*Mxy)/det/2.
    Ycenter = (Myz*(Mxx - x) - Mxz*Mxy)/det/2.

    x = Xcenter + X.mean()
    y = Ycenter + Y.mean()
    r = sqrt(abs(Xcenter**2 + Ycenter**2 + Mz))
    s = sigma(coords,x,y,r)
    iter_ = i
    if verbose:
        print('Regression complete in {} iterations.'.format(iter_))
        print('Sigma computed: ', s)
    return x, y, r, s


def create_circle_tree(topics: list, n_words: int = 10):

    if any([len(t) > n_words for t in topics]):
        new_topics = [t[:10] for t in topics]
        topics = new_topics

    _, ax = plt.subplots(figsize=(40, 22))
    graph = nx.Graph()

    norm = mpl.colors.Normalize(vmin=0, vmax=len(topics) + 2, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap('tab20'))
    node_colors = []

    graph.add_nodes_from([(" ", {"topic": -1})])

    node_colors.append(mapper.to_rgba(0))

    topic_id_to_nodes = {}
    for i_t, topic in enumerate(topics):

        topic_name = "Topic " + str(i_t + 1)

        graph.add_nodes_from([(topic_name, {"topic": i_t + 1})])
        node_colors.append(mapper.to_rgba(i_t + 1))


        topic_nodes = []
        for w in topic:

            if w in graph.nodes():
                not_added = True

                while not_added:

                    w += " "
                    if w not in graph.nodes():
                        graph.add_nodes_from([(w, {"topic": i_t + 1})])
                        not_added = False
            else:
                graph.add_nodes_from([(w, {"topic": i_t + 1})])

            topic_nodes.append(w)

            node_colors.append(mapper.to_rgba(i_t + 1))

            graph.add_edge(w, topic_name)

            # node_num += 1

        topic_id_to_nodes.update({i_t: topic_nodes})
        graph.add_edge(topic_name, " ")

    # assert len(topic_nums) == len(topics)

    pos = nx.circular_layout(graph, scale=1)
    pos[" "] = [0, 0]
    for i_t in range(len(topics)):

        topic_poses = [pos[" "]]
        for n in topic_id_to_nodes[i_t]:
            topic_poses.append(pos[n])

        t_x, t_y, _, _ = hyper_fit(topic_poses)

        pos["Topic " + str(i_t + 1)] = [t_x, t_y]

    nx.draw(graph, pos=pos, node_size=0, linewidths=0.01,
            font_size=15, with_labels=False, ax=ax)
    labels_pos = nx.draw_networkx_labels(graph, pos=pos, font_size=20, horizontalalignment='center',
                                         verticalalignment='center', ax=ax)

    topic_orientation = ''
    topic_alignment = ''
    i_n = 0
    for n, t in labels_pos.items():
        t.set_color(node_colors[i_n])
        i_n += 1

        # check if root node
        if n == " ":
            continue

        # check if topic node:
        if "Topic " in n:
            x, y = pos[n]

            # define topic orientation
            if abs(x) > abs(y):
                topic_orientation = 'horizontal'

                if x > 0:
                    topic_alignment = 'left'
                else:
                    topic_alignment = 'right'
            else:
                topic_orientation = 'vertical'

                if y > 0:
                    topic_alignment = 'bottom'
                else:
                    topic_alignment = 'top'

        else:
            # word node
            # apply topic orientation and alignment to all its word nodes
            t.set_rotation(topic_orientation)

            if topic_orientation == 'horizontal':
                t.set_horizontalalignment(topic_alignment)
            else:
                t.set_verticalalignment(topic_alignment)

    return graph, plt


def scatter_plot(x: list, ys: list, x_label: str, y_label: str, color_legends: list, type: str) -> plt:
    fig, ax = vis_prep()

    assert type in ["c_v", "u_mass", "c_npmi", "dbs"], "define the type of scatter plot"
    assert isinstance(ys[0], list), "ys needs to be a list of list(s)"
    assert len(ys) == len(color_legends), "need a color legend for each y list (ys)"

    mapper = mpl.cm.get_cmap('Pastel2')
    ys_color = [mapper(i_y) for i_y,_ in enumerate(ys)]

    for i_y, y in enumerate(ys):
        plt.plot(x, y, 'o-', c=ys_color[i_y], markersize=17, linewidth=3, label=color_legends[i_y])

    if type == "u_mass":
        y_ticks = [x for x in range(-6, 8, 2)]
    if type == "c_npmi":
        # todo: set dynamically with min and max
        y_ticks = [x/10 for x in range(-1, 6, 1)]
    elif type == "c_v":
        y_ticks = [x / 10 for x in range(0, 11, 1)]
    elif type == "dbs":
        all_y = []
        for y in ys:
            all_y.extend((y))
        y_ticks = [x/10 for x in range(10, 40, 5)]

    ax.set_xlabel(x_label, fontsize='medium')
    ax.set_ylabel(y_label, fontsize='medium')

    ax.yaxis.set_ticks(y_ticks)
    ax.xaxis.set_ticks(x)

    ax.tick_params(axis='both', labelsize='small')

    plt.legend(fontsize='x-small')
    plt.grid(color='grey', axis='y', linestyle='--', linewidth=0.7)
    return plt, fig


def write_topics(topics: list, filename: str, n_words: int = 10):
    if any([len(t) > n_words for t in topics]):
        new_topics = [t[:10] for t in topics]
        topics = new_topics

    with open(filename, "w") as myFile:

        for i, t in enumerate(topics):
            myFile.write('Topic ' + str(i+1) + '\n')

            for w in t:
                myFile.write(w + ' ')

            myFile.write('\n\n\n')


def box_plot(ys: list, labels: list, x_label: str, y_label: str):

    assert isinstance(ys[0], list), "ys needs to be a list of list(s)"
    assert len(ys) == len(labels), "need label for each y list (ys)"

    fig, ax = vis_prep()

    mean_line_color = 'firebrick'
    mean_line_width = 2

    median_props = dict(linestyle='-', linewidth=0, color='white')
    mean_line_props = dict(linestyle='-', linewidth=mean_line_width, color=mean_line_color)

    plt.boxplot(ys, labels=labels, showmeans=True, meanline=True, meanprops=mean_line_props, medianprops=median_props)

    ax.set_xlabel(x_label, fontsize='medium')
    ax.set_ylabel(y_label, fontsize='medium')

    ax.tick_params(axis='both', labelsize='x-small')

    mean_line_legend = [Line2D([0], [0], color=mean_line_color, linewidth=mean_line_width, linestyle='-')]
    plt.legend(mean_line_legend, ["mean"], fontsize='x-small')

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


if __name__ == "__main__":
    # create_circle_tree([["word", "sadsa", "sdadas"], ["sadsa2", "eada", "1431324", "sasa"],
    #                    ["aa", "tt"], ["adsada", "y"]])

    # plt, fig = scatter_plot([2, 4, 6], [[0.2, 0.5, 0.5], [0.4, 0.3, 0.6]], x_label="x axis", y_label="y axis",
    #                    color_legends=["Topic 1", "Topic 2"], type='c_v')


    #plt = box_plot([0, 1], [[0.2, 0.5, 0.5], [0.4, 0.3, 0.6]], ["1", "2"], "x label", "y label")
    # plt.show()
    #fig.savefig("visuals/cs_baseline_vs_k.pdf", bbox_inches='tight', transparent=True)

    # write_topics([["this", "that", "those"], ["cat", "dog", "ant"]], "text.txt")

    """G = nx.cycle_graph(80)
    pos = nx.circular_layout(G)
    pylab.figure(1)
    nx.draw(G, pos)
    pylab.figure(2)
    nx.draw(G, pos, node_size=60, font_size=8)
    pylab.figure(3, figsize=(12, 12))
    nx.draw(G, pos)
    pylab.show()"""
