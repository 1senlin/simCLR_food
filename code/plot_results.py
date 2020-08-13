from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
tsne = TSNE()


def plot_tsne(y, y_labels):
    trans=tsne.fit_transform(y)
    sns.scatterplot(trans[:,0],trans[:,1], hue=y_labels)
    #plt.legend(list(set(y_labels)))
    plt.savefig('test.jpg')
    plt.close()