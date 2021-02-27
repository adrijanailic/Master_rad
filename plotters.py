from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Plotter:

    def __init__(self, title=''):
        self.title = title
        
    def scatter(self, X, y):
        plt.figure()
        plt.grid(True)
        plt.title(self.title)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Accent')
        plt.show()
        
    def plot(self, X, y):
        plt.figure()
        plt.grid(True)
        plt.title(self.title)
        plt.plot(X, y)  
        plt.show()
        
    def plot_losses(self, history, epochs):
        train_loss = history.history['loss']
        val_loss   = history.history['val_loss']
        xc         = range(epochs)
        
        plt.figure()
        plt.grid(True)
        plt.title('Losses')
        plt.plot(xc, train_loss, label='train loss')
        plt.plot(xc, val_loss, label='validation loss')
        plt.xlabel('epochs')
        plt.legend(loc='best')
        plt.show()
        
    def pca_plot(self, X, y, embedding_model):
        X_embedded = embedding_model.predict(X)
        pca_original = PCA(n_components=2).fit_transform(X)
        pca_embedded = PCA(n_components=2).fit_transform(X_embedded)
        
        plt.figure(figsize=(9, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.set_title('Original data')
        ax2.set_title('Embedded data')
        ax1.grid()
        ax2.grid()
        ax1.scatter(pca_original[:, 0], pca_original[:, 1], c=y, cmap='Accent')
        ax2.scatter(pca_embedded[:, 0], pca_embedded[:, 1], c=y, cmap='Accent')
        plt.show()
