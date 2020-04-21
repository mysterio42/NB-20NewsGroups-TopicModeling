import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

FIGURES_DIR = 'figures/'


def plot_optimal_alpha(alpha_range, scores, opt_alpha, opt_score):
    fig, ax = plt.subplots()
    ax.plot(alpha_range, scores)
    ax.plot(opt_alpha, opt_score, 'ro', color='green', markersize=10)
    ax.annotate('   optimal alpha', (opt_alpha, opt_score))
    ax.set(xlabel='Possible alpha values', ylabel='Accuracy',
           title=f'Optimal alpha: {opt_alpha} Optimal Score: {opt_score}')
    plt.savefig(FIGURES_DIR + 'Figure_opt_alpha' + '.png')
    plt.show()


def plot_cm(cm):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.savefig(FIGURES_DIR + f'Figure_cm' + '.png')
    plt.show()


def plot_pca(features, labels, name):
    cmap = colors.ListedColormap(['blue', 'red', 'green', 'yellow'])
    bounds = [0, 5, 10]
    colors.BoundaryNorm(bounds, cmap.N)

    plt.figure()
    plt.title(label=f'PCA on the {name} data')
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap=cmap)
    plt.colorbar()
    plt.savefig(FIGURES_DIR + f'Figure_pca_{name}' + '.png')
    plt.show()


def plot_tsne(features, labels, name):
    cmap = colors.ListedColormap(['blue', 'red', 'green', 'yellow'])
    bounds = [0, 5, 10]
    colors.BoundaryNorm(bounds, cmap.N)

    fig = figure()
    ax = Axes3D(fig)

    tsne = TSNE(n_components=3)
    Z = tsne.fit_transform(features)
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=labels, cmap=cmap)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(f'T-SNE on the {name} data')
    plt.savefig(FIGURES_DIR + f'Figure_tsne_{name}' + '.png')
    plt.show()
