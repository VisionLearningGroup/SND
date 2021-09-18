import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import ListedColormap
from DANN import DANN
from sklearn.datasets import make_gaussian_quantiles

from matplotlib import pyplot
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from scipy.spatial.distance import cdist


def softmax( z):
    """
    Softmax function.

    """
    v = np.exp(z)
    return v / np.sum(v, axis=0)

def entropy(mat, prob=True):
    if prob:
        prob = softmax(mat)
    else:
        prob = mat
    return np.sum(-prob * np.log(prob), axis=1).mean()


def rot_theta(X, theta):
    from math import cos, sin, pi
    trans = -np.mean(X, axis=0)
    #X = 2 * (X + trans)
    X = 2 * (X + trans)
    theta = -theta * pi / 180
    rotation = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    X = np.dot(X, rotation.T)
    return X

def main(cov=1.0, rand=1, dann=False):
    cov = cov
    X1, y1 = make_gaussian_quantiles(cov=cov,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=rand)


    X2, y2 = make_gaussian_quantiles(mean=(5, 5), cov=cov,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=rand+1)


    Xt1, yt1 = make_gaussian_quantiles(cov=cov,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=rand+2)

    Xt2, yt2 = make_gaussian_quantiles(mean=(5, 5), cov=cov,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=rand+3)
    ## shift points
    Xt1 += 1.5
    y1 = np.zeros(y1.shape[0])
    y2 = np.ones(y2.shape[0])
    yt1 = np.zeros(yt1.shape[0])
    #yt2 = np.ones(yt2.shape[0])
    Xall = np.vstack((X1, X2))
    Xtall = Xt1
    label_all = np.r_[y1, y2]
    labelt_all = yt1
    # NN training
    algo = DANN(hidden_layer_size=15, maxiter=500,
                lambda_adapt=1., seed=42, adversarial_representation=dann)
    algo.fit(Xall, label_all, Xtall)
    result = algo.forward(Xtall).T
    pred = algo.predict(Xtall)
    correct = (pred == labelt_all)
    acc = correct.sum() / float(len(correct))
    normed = result / np.linalg.norm(result,axis=1).reshape(result.shape[0], 1)
    range_sample = [5, 10, 20, 30, 40, 50, 100, 250, 300]
    for sample in range_sample:
        random_v = np.random.permutation(len(normed))
        norm_com = normed[random_v[:sample]]
        mat = np.dot(norm_com, norm_com.T) #/ 0.05
        print("sim ent %s" % entropy(mat))
    print("acc %s"%acc)
    print("sim ent %s"%entropy(mat))
    print("entropy %s"%entropy(result, prob=False))
    pyplot.title("Variance %s Similarity Entropy %.3f Accuracy %.3f"%(cov, entropy(mat), acc))
    print(Xtall.shape)
    draw_trans_data(Xall, label_all, Xtall, algo.predict, dann=dann, algo=algo)
    pyplot.show()
    pyplot.savefig("boundary_%s_dann_%s.png"%(cov, dann))
    pyplot.clf()
    x = algo.forward(Xtall).T[:, 1]
    fig, ax = pyplot.subplots(tight_layout=True)
    pyplot.ylim((0, 200))
    n_bins = 20
    ax.hist(x, bins=n_bins)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    pyplot.xlabel('Class probability of a circle class', fontsize=14)
    pyplot.savefig("hist_%s_dann_%s.png"%(cov, dann))
    pyplot.clf()
    X_hidden = algo.hidden_representation(Xall)
    Xt_hidden = algo.hidden_representation(Xtall)
    hidden_all = np.vstack((X_hidden, Xt_hidden))
    X_embedded = TSNE(n_components=2, perplexity=30).fit_transform(hidden_all)
    x_min, x_max = np.min(X_embedded, 0), np.max(X_embedded, 0)
    X_embedded = (X_embedded - x_min) / (x_max - x_min)
    ax = pyplot.subplot(111)
    for i in range(len(label_all)):
        if label_all[i] == 1:
            pyplot.scatter(X_embedded[i, 0], X_embedded[i, 1], c='g', marker='o', s=40)
        else:
            pyplot.scatter(X_embedded[i, 0], X_embedded[i, 1], c='r', marker='>', s=40)
    X_embedded_t = X_embedded[X_hidden.shape[0]:]
    pyplot.scatter(X_embedded_t[:, 0], X_embedded_t[:, 1], c='k', marker='>', s=40)
    ax.tick_params(labelbottom="off", bottom="off")
    ax.tick_params(labelleft="off", left="off")
    pyplot.tick_params(color='white')
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    pyplot.savefig("tsne_hidden_%s_dann_%s.png" % (cov, dann))
    pyplot.clf()
    draw_trans_data_v2(Xall, label_all, Xtall, algo.predict, dann=dann, algo=algo)
    pyplot.savefig("plot_inputspace.png")

    return x


def make_trans_moons(theta=40, nb=100, noise=.05):
    from math import cos, sin, pi

    X, y = make_moons(nb, noise=noise, random_state=1)
    Xt, yt = make_moons(nb, noise=noise, random_state=2)

    trans = -np.mean(X, axis=0)
    X = 2 * (X + trans)
    Xt = 2 * (Xt + trans)

    theta = -theta * pi / 180
    rotation = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    Xt = np.dot(Xt, rotation.T)

    return X, y, Xt, yt


def draw_trans_data(X, y, Xt, predict_fct=None, neurons_to_draw=None, colormap_index=0, special_points=None,
                    special_xytext=None, dann=False, algo=None):
    # Some line of codes come from: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    if colormap_index == 0:
        cm_bright = ListedColormap(['#FF0000', '#00FF00'])
    else:
        cm_bright = ListedColormap(['#0000FF', '#000000'])

    x_min, x_max = 1.1 * X[:, 0].min(), 1.1 * X[:, 0].max()
    y_min, y_max = 1.1 * X[:, 1].min(), 1.1 * X[:, 1].max()

    pyplot.xlim((x_min, x_max))
    pyplot.ylim((y_min, y_max))

    pyplot.tick_params(direction='in', labelleft=False)

    if predict_fct is not None:
        h = .02  # step size in the mesh

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = predict_fct(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        pyplot.contourf(xx, yy, Z, cmap=cm_bright, alpha=.4)
        pyplot.contour(xx, yy, Z, colors='black', linewidths=5)

    if X is not None:
        for i in range(len(y)):
            if y[i] == 1:
                pyplot.scatter(X[i, 0], X[i, 1], c='g', marker='o', s=40)
            else:
                pyplot.scatter(X[i, 0], X[i, 1], c='r', marker='>', s=40)

    if Xt is not None:
        pyplot.scatter(Xt[:, 0], Xt[:, 1], c='k', marker='>', s=40)

    if special_points is not None:
        for i in range(np.shape(special_points)[0]):
            if special_xytext is None:
                xytext = (30, 45) if i % 2 == 1 else (-40, -60)
            else:
                xytext = special_xytext[i]

            pyplot.annotate('ABCDEFG'[i], special_points[i, :], xycoords='data', color="blue",
                            xytext=xytext, textcoords='offset points',
                            size=32,
                            arrowprops=dict(arrowstyle="fancy", fc=(0., 0., 1.), ec="none",
                                            connectionstyle="arc3,rad=0.0"))

    if neurons_to_draw is not None:
        for w12, b in zip(neurons_to_draw[0], neurons_to_draw[1]):
            w1, w2 = w12
            get_y = lambda x: -(w1 * x + b) / w2
            pyplot.plot([x_min, x_max], [get_y(x_min), get_y(x_max)])



def draw_trans_data_v2(X, y, Xt, predict_fct=None, neurons_to_draw=None, colormap_index=0, special_points=None,
                    special_xytext=None, dann=False, algo=None):
    # Some line of codes come from: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    #X = algo.forward(X).T
    #Xt = algo.forward(Xt).T
    x_min, x_max = 1.1 * X[:, 0].min(), 1.1 * X[:, 0].max()
    y_min, y_max = 1.1 * X[:, 1].min(), 1.1 * X[:, 1].max()
    ax = pyplot.subplot(111)
    pyplot.xlim((x_min, x_max))
    pyplot.ylim((y_min, y_max))
    pyplot.tick_params(direction='in', labelleft=False)
    if X is not None:
        for i in range(len(y)):
            if y[i] == 1:
                pyplot.scatter(X[i, 0], X[i, 1], c='g', marker='o',s=40)
            else:
                pyplot.scatter(X[i, 0], X[i, 1], c='r',  marker='>',s=40)
    if Xt is not None:
        pyplot.scatter(Xt[:, 0], Xt[:, 1], c='k',  marker='>',s=40)
    ax.tick_params(labelbottom="off", bottom="off")
    ax.tick_params(labelleft="off", left="off")
    pyplot.tick_params(color='white')
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")


def run_pca(X, y, Xt, algo, special_points=None, special_xytext=None, mult=None):
    if mult is None:
        mult = np.ones(2)

    h_X = algo.hidden_representation(X)
    h_Xt = algo.hidden_representation(Xt)

    pca = PCA(n_components=2)
    pca.fit(np.vstack((h_X, h_Xt)))

    if special_points is not None:
        special_points = mult * pca.transform(algo.hidden_representation(special_points))

    draw_trans_data(mult * pca.transform(h_X), y, mult * pca.transform(h_Xt), special_points=special_points,
                    special_xytext=special_xytext)


if __name__ == '__main__':
    cov = 0.5
    result_nodann = main(cov, 0, False)
    result_dann = main(cov, 0, True)
    n_bins = 20

