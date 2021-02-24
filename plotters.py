from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PCAPlotter:

    def __init__(self, X, y, embedding_model):
        self.embedding_model = embedding_model
        self.X = X
        self.y = y
        self.X_embedded = self.embedding_model.predict(self.X)
        plt.figure(figsize=(9, 4))
        plt.grid(True)
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax2 = plt.subplot(1, 2, 2)
        self.ax1.set_title('Original data')
        self.ax2.set_title('Embedded data')
        self.ax1.grid()
        self.ax2.grid()

    def plot(self):
        pca_original_out = PCA(n_components=2).fit_transform(self.X)
        pca_embedded_out = PCA(n_components=2).fit_transform(self.X_embedded)
        self.ax1.scatter(pca_original_out[:, 0], pca_original_out[:, 1], c=self.y, cmap='Accent')
        self.ax2.scatter(pca_embedded_out[:, 0], pca_embedded_out[:, 1], c=self.y, cmap='Accent')
        plt.show()


# Use this class to scatter X data.
class Plotter:

    # TODO Maybe add title as a member?
    def __init__(self, X, y):
        self.X = X
        self.y = y
        plt.figure(figsize=(9, 4))
        plt.grid(True)
        plt.title('Data')

    def plot(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='Accent')
        plt.show()
