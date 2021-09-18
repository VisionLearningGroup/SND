import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from DANN import DANN
from sklearn.datasets import make_gaussian_quantiles, make_classification, make_blobs

from matplotlib import pyplot


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
def main(nc, rand=1):

    cov = 1.0
    #for i in range(nc):
    X1, y1 = make_blobs(n_features=2, centers=nc, n_samples=1000, random_state=1)

    Xall = X1[:500]#np.vstack((X1, X2))
    Xtall = X1[500:]#np.vstack((Xt1, Xt2))
    label_all = y1[:500]#np.r_[y1, y2]
    labelt_all = y1[500:]#np.r_[yt1, yt2]#np.vstack((yt1, yt2))
    special = np.array([[-2.4, -1.6], [-1.2, 0.4], [.8, -.5], [2.5, 1.5]])
    special_points = Xall[np.argmin(cdist(special, Xall), axis=1), :]
    # Standard NN
    algo = DANN(hidden_layer_size=15, maxiter=500, lambda_adapt=6., seed=42, adversarial_representation=False)
    algo.fit(Xall, label_all, Xtall)
    result = algo.forward(Xtall).T
    pred = algo.predict(Xtall)
    correct = (pred == labelt_all)
    acc = correct.sum() / float(len(correct))
    normed = result / np.linalg.norm(result,axis=1).reshape(result.shape[0], 1)
    mat = np.dot(normed, normed.T) / 0.05
    print("acc %s"%acc)
    print("sim ent %s"%entropy(mat))
    print("entropy %s"%entropy(result, prob=False))
    draw_trans_data(Xtall, labelt_all, None, algo.predict)

    pyplot.show()
    pyplot.savefig("nc_%s.png"%(nc))
    pyplot.clf()


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
                    special_xytext=None):
    # Some line of codes come from: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    N = 20
    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colorst = [colormap(i) for i in np.linspace(0, 0.9, N)]
    if colormap_index == 0:
        cm_bright = ListedColormap(['#FF0000', '#00FF00'])
    else:
        cm_bright = ListedColormap(['#0000FF', '#000000'])

    x_min, x_max = -15, 2#1.1 * X[:, 0].min(), 1.1 * X[:, 0].max()
    y_min, y_max = 1.1 * X[:, 1].min(), 1.1 * X[:, 1].max()

    pyplot.xlim((x_min, x_max))
    pyplot.ylim((y_min, y_max))

    pyplot.tick_params(direction='in', labelleft=False)

    if X is not None:
        for i in range(len(y)):
            if y[i] == 1:
                pyplot.scatter(X[i, 0], X[i, 1], c=colorst[y[i]], s=40)
            else:
                pyplot.scatter(X[i, 0], X[i, 1], c=colorst[y[i]], s=40)

    if Xt is not None:
        pyplot.scatter(Xt[:, 0], Xt[:, 1], c='k', s=40)

def run_pca(X, y, Xt, algo, special_points=None, special_xytext=None, mult=None):
    if mult is None:  # mult is used to flip the representation
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
    #covs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    num_classes = range(5, 30)
    #randoms = range(len(covs))
    #for rand in randoms:
    for nc in num_classes:
        print("nc %s"%nc)
        main(nc, 0)
