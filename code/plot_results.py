from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
tsne = TSNE(perplexity=100)


def plot_tsne(y, y_labels):
    if y.shape[-1]>50:
        pca=PCA(n_components=40)
        y=pca.fit_transform(y)
        print(y.shape)
        
    trans=tsne.fit_transform(y)
    sns.scatterplot(trans[:,0],trans[:,1], hue=y_labels)
    #plt.legend(list(set(y_labels)))
    plt.savefig('test.jpg')
    plt.close()