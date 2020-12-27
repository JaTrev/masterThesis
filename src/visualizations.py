import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import sqrt
from sklearn.manifold import TSNE


plt.figure(figsize=(10, 8))


def sigma(coords, x, y, r):
    """Computes Sigma for circle fit."""
    dx, dy, sum_ = 0., 0., 0.

    for i in range(len(coords)):
        dx = coords[i][1] - x
        dy = coords[i][0] - y
        sum_ += (sqrt(dx*dx+dy*dy) - r)**2
    return sqrt(sum_/len(coords))


def hyper_fit(coords, IterMax=99, verbose=False):
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


def create_circle_tree(topics: list):

    G = nx.Graph()

    graph_label = " "
    G.add_node(graph_label)
    for i_t, topic in enumerate(topics):

        topic_label = "Topic " + str(i_t + 1)
        G.add_node(topic_label)

        for w in topic:

            G.add_node(w)
            G.add_edge(w, topic_label)

        G.add_edge(topic_label, graph_label)

    pos = nx.circular_layout(G, scale=2)
    pos[graph_label] = [0, 0]
    for i_t, topic in enumerate(topics):
        topic_label = "Topic " + str(i_t + 1)

        topic_pos = [pos[graph_label]]
        topic_pos.extend([pos[w] for w in topic])

        t_x, t_y, _, _ = hyper_fit(topic_pos)

        pos[topic_label] = [t_x, t_y]

    norm = mpl.colors.Normalize(vmin=0, vmax=len(topics) + 2, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gist_ncar)

    node_colors = []
    for n in G.nodes:

        if n == " ":
            node_colors.append(mapper.to_rgba(0))
            continue

        found_topic = False
        for i_t, topic in enumerate(topics):

            if n in topic:
                node_colors.append(mapper.to_rgba(i_t + 1))
                found_topic = True
                break

        if not found_topic:
            topic_num = int(''.join(x for x in n if x.isdigit()))
            # print("topic_num: " + str(topic_num))

            node_colors.append(mapper.to_rgba(int(topic_num)))

    assert len(node_colors) == len(G.nodes)

    nx.draw(G, pos=pos, node_size=100, node_color=node_colors, linewidths=0.01,
            font_size=15, with_labels=True, )

    plt.show()


def scatter_plot(x: list, ys: list) -> plt:

    assert isinstance(ys[0], list), "ys needs to be a list of list(s)"

    norm = mpl.colors.Normalize(vmin=0, vmax=len(ys) + 1, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gist_ncar)
    ys_color = [mapper.to_rgba(i_y) for i_y,_ in enumerate(ys)]

    for i_y, y in enumerate(ys):
        plt.plot(x, y, 'o', c=ys_color[i_y], markersize=20, linewidth=5)

    # plt.show()
    return plt


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
    create_circle_tree([["word", "sadsa", "sdadas"], ["sadsa2", "eada", "1431324", "sasa"],
                        ["aa", "tt"], ["adsada", "y"]])
