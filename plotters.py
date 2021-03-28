from sklearn.decomposition import PCA
from SiameseNet import SiameseNet
from TripletNet import TripletNet
import matplotlib.pyplot as plt

# Plotter class.
class Plotter:   
    # Scatter data.
    # X     - Data to be scattered.
    # y     - Data labels.
    # title - Optional plot title to be displayed.
    def scatter(self, X, y, title=''):
        pca_x = PCA(n_components=2).fit_transform(X)
        
        plt.figure()
        plt.grid(True)
        plt.title(title)
        plt.scatter(pca_x[:, 0], pca_x[:, 1], c=y, cmap='Accent')
        plt.show()
        
    # Plot data.
    # X, y  - Data to be plotted.
    # title - Optional plot title to be displayed.
    def plot(self, X, y, title=''):
        plt.figure()
        plt.grid(True)
        plt.title(title)
        plt.plot(X, y)  
        plt.show()
        
    # Plots losses based on history object produced during training process.
    # net     - Net that was trained.
    # history - Object produced during training process.
    def plot_losses(self, net, history):
        train_loss = history.history['loss']
        val_loss   = history.history['val_loss']
        xc         = range(net.epochs)
        
        if isinstance(net, SiameseNet):
            folder_name = 'contrastive'
            model_name = 'HISTORY_model' + str(net.model_handler.model_number) \
                + '_alpha'     + str(net.alpha) \
                + '_epochs'    + str(net.epochs) \
                + '_batchSize' + str(net.batch_size) \
                + '_steps'     + str(net.steps_per_epoch)
        elif isinstance(net, TripletNet):
            folder_name = 'triplet'
            if net.mining_method == 'create_triplet_batch_random':
                model_name = 'HISTORY_model' + str(net.model_handler.model_number) \
                           + '_mining-'   + str(net.mining_method) \
                           + '_alpha'     + str(net.alpha) \
                           + '_epochs'    + str(net.epochs) \
                           + '_batchSize' + str(net.batch_size) \
                           + '_steps'     + str(net.steps_per_epoch)
            else:
                model_name = 'HISTORY_model' + str(net.model_handler.model_number) \
                           + '_mining-'   + str(net.mining_method) \
                           + '_alpha'     + str(net.alpha) \
                           + '_epochs'    + str(net.epochs) \
                           + '_numberOfSamplesPerClass' + str(net.number_of_samples_per_class) \
                           + '_steps'     + str(net.steps_per_epoch)
        
        plt.figure()
        plt.grid(True)
        plt.title('Losses')
        plt.plot(xc, train_loss, label='train loss')
        plt.plot(xc, val_loss, label='validation loss')
        plt.xlabel('epochs')
        plt.legend(loc='best')
            
        plt.savefig('pics/' + folder_name + '/' + model_name + '.png', bbox_inches="tight")
        plt.show()
        
    # # Set parameters for plotting.
    # # workspace - Dictionary object with parameters values.
    # def set_plot_parameters(self, workspace):
    #     self.net_type = workspace["net"]
    #     self.model_number = workspace["model_number"]
    #     self.batch_size = workspace["batch_size"]
    #     self.epochs = workspace["epochs"]
    #     self.steps_per_epoch = workspace["steps_per_epoch"]
    #     self.alpha = workspace["alpha"]
    #     self.mining_method = workspace["mining_method"]
    #     self.number_of_samples_per_class = workspace["number_of_samples_per_class"]
        
    # Make 2 PCA subplots - one with original data, and one with the same data, embedded with net.
    # X        - Data to be plotted.
    # y        - Labels of X data.
    # net      - Net with which to embed X data.
    # suptitle - Optional suptitle for the produced figure.
    def pca_plot(self, X, y, net, suptitle=''):
        X_embedded = net.embedding_model.predict(X)
        pca_original = PCA(n_components=2).fit_transform(X)
        pca_embedded = PCA(n_components=2).fit_transform(X_embedded)
        
        fig = plt.figure(figsize=(9, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
        ax1.set_title('Original data')
        ax2.set_title('Embedded data')
        ax1.grid()
        ax2.grid()
        ax1.scatter(pca_original[:, 0], pca_original[:, 1], c=y, cmap='Accent')
        ax2.scatter(pca_embedded[:, 0], pca_embedded[:, 1], c=y, cmap='Accent')
        
        if isinstance(net, SiameseNet):
            folder_name = 'contrastive'
            figtext = "embedding size = " + str(net.embedding_size) + " \n" \
                  + "alpha = " + str(net.alpha) + " \n" \
                  + "batch size = " + str(net.batch_size) + " \n" \
                  + "epochs = " + str(net.epochs) + " \n" \
                  + "steps per epoch = " + str(net.steps_per_epoch)
            
            model_name = suptitle + '_model' + str(net.model_handler.model_number) \
                  + '_alpha' + str(net.alpha) \
                  + '_epochs' + str(net.epochs) \
                  + '_batchSize' + str(net.batch_size) \
                  + '_steps' + str(net.steps_per_epoch)
        
        elif isinstance(net, TripletNet):
            folder_name = 'triplet'
            if net.mining_method == "create_triplet_batch_random":
                figtext = "mining method = " + net.mining_method + " \n" \
                      + "embedding size = "  + str(net.embedding_size) + " \n" \
                      + "alpha = "           + str(net.alpha) + " \n" \
                      + "batch size = "      + str(net.batch_size) + " \n" \
                      + "epochs = "          + str(net.epochs) + " \n" \
                      + "steps per epoch = " + str(net.steps_per_epoch)
                      
                model_name = suptitle + '_model' + str(net.model_handler.model_number) \
                           + '_mining-'   + str(net.mining_method) \
                           + '_alpha'     + str(net.alpha) \
                           + '_epochs'    + str(net.epochs) \
                           + '_batchSize' + str(net.batch_size) \
                           + '_steps'     + str(net.steps_per_epoch)
            else:
                figtext = "mining method = " + net.mining_method + " \n" \
                      + "embedding size = "  + str(net.embedding_size) + " \n" \
                      + "alpha = "           + str(net.alpha) + " \n" \
                      + "number of samples per class = " + str(net.number_of_samples_per_class) + " \n" \
                      + "epochs = "          + str(net.epochs) + " \n" \
                      + "steps per epoch = " + str(net.steps_per_epoch)
                      
                model_name = suptitle + '_model' + str(net.model_handler.model_number) \
                           + '_mining-'   + str(net.mining_method) \
                           + '_alpha'     + str(net.alpha) \
                           + '_epochs'    + str(net.epochs) \
                           + '_numberOfSamplesPerClass' + str(net.number_of_samples_per_class) \
                           + '_steps'     + str(net.steps_per_epoch)
                
        plt.figtext(0.15, -0.2, figtext, ha="left", fontsize=10, bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
            
        plt.savefig('pics/' + folder_name + '/' + model_name + '.png', bbox_inches="tight")
        plt.show()
