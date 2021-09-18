import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.colors import ListedColormap
from DANN import DANN
from sklearn.datasets import make_gaussian_quantiles

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
def main(cov=1.0, margin=1.0, rand=1):
    cov = cov

    X1, y1 = make_gaussian_quantiles(cov=cov,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=rand)


    X2, y2 = make_gaussian_quantiles(mean=(margin, margin), cov=cov,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=rand+1)
    Xt1, yt1 = make_gaussian_quantiles(cov=cov,
                                     n_samples=300, n_features=2,
                                     n_classes=2, random_state=rand+2)


    Xt2, yt2 = make_gaussian_quantiles(mean=(margin, margin), cov=cov,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=rand+3)
    import pdb
    y1 = np.zeros(y1.shape[0])
    y2 = np.ones(y2.shape[0])
    yt1 = np.zeros(yt1.shape[0])
    yt2 = np.ones(yt2.shape[0])
    #pdb.set_trace()
    #X, y, Xt, yt = make_trans_moons(35, nb=150)
    Xall = np.vstack((X1, X2))
    Xtall = np.vstack((Xt1, Xt2))
    label_all = np.r_[y1, y2]
    labelt_all = np.r_[yt1, yt2]#np.vstack((yt1, yt2))
    #pdb.set_trace()
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
    #import pdb
    #pdb.set_trace()
    print("acc %s"%acc)
    print("sim ent %s"%entropy(mat))
    print("entropy %s"%entropy(result, prob=False))

    #pyplot.subplot(2, 4, 1)
    pyplot.title("Variance %s Similarity Entropy %.3f Accuracy %.3f"%(cov, entropy(mat), acc))
    draw_trans_data(Xtall, labelt_all, None, algo.predict)

    # pyplot.subplot(2, 4, 2)
    # pyplot.title("NN: Representation PCA")
    # run_pca(X, y, Xt, algo, special_points=special_points, mult=[-1, -1])
    #
    # pyplot.subplot(2, 4, 3)
    # pyplot.title("NN: Domain classification")
    # draw_trans_data(X, y, Xt, algo.predict_domain, colormap_index=1)
    #
    # pyplot.subplot(2, 4, 4)
    # pyplot.title("NN: Hidden neurons")
    # draw_trans_data(X, y, Xt, neurons_to_draw=(algo.W, algo.b))
    #
    # # DANN
    # algo = DANN(hidden_layer_size=15, maxiter=500, lambda_adapt=6., seed=42)
    # algo.fit(X, y, Xt)
    #
    # pyplot.subplot(2, 4, 5)
    # pyplot.title("DANN: Label classification")
    # draw_trans_data(X, y, Xt, algo.predict, special_points=special_points,
    #                 special_xytext=[(50, -15), (-20, -90), (-50, 40), (-80, 0)])
    #
    # pyplot.subplot(2, 4, 6)
    # pyplot.title("DANN: Representation PCA")
    # run_pca(X, y, Xt, algo, special_points=special_points, mult=[-1, 1],
    #         special_xytext=[(-10, -80), (50, -60), (-40, 50), (-20, 70)])
    #
    # pyplot.subplot(2, 4, 7)
    # pyplot.title("DANN: Domain classification")
    # draw_trans_data(X, y, Xt, algo.predict_domain, colormap_index=1)
    #
    # pyplot.subplot(2, 4, 8)
    # pyplot.title("DANN: Hidden neurons")
    # draw_trans_data(X, y, Xt, neurons_to_draw=(algo.W, algo.b))

    pyplot.show()
    pyplot.savefig("tmp_%s.png"%(cov))
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
                pyplot.scatter(X[i, 0], X[i, 1], c='g', s=40)
                #pyplot.annotate("o", X[i, :], color="green", size=50 * 1.5, textcoords='offset points',
                #                xytext=(-6 * 1.5, -13 * 1.5))
            else:
                pyplot.scatter(X[i, 0], X[i, 1], c='r', s=40)
                #pyplot.annotate("o", X[i, :], color="red", size=30 * 1.5, textcoords='offset points',
                #                xytext=(-8 * 1.5, -8 * 1.5))

    if Xt is not None:
        pyplot.scatter(Xt[:, 0], Xt[:, 1], c='k', s=40)

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
    margins = np.linspace(1.5, 3.5, num=10)
    randoms = range(len(margins))
    for rand in randoms:
        for margin in margins:
            print("margin %s"%margin)
            main(0.5, margin, rand)
