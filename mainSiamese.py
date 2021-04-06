from SiameseNet import SiameseNet
from ModelHandler import ModelHandler
from DataHandler import DataHandler
from plotters import Plotter

# %% MNIST dataset
dh = DataHandler("MNIST", classes_to_select=[0,1,2,3,4,5,6])
#dh_newdata = DataHandler("MNIST", classes_to_select=[7,8,9])

# %% Define embedding model
mh = ModelHandler(model_number=4, embedding_size=200, input_feature_dim=dh.shape)

# %% Define siamese net
#alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
alphas = [0.1]
for alpha in alphas:
    net = SiameseNet(mh, dh, alpha)
    net.print_model()
    batch_size = 200
    epochs = 1
    steps_per_epoch = 1#int(dh.n_train / batch_size)
    history = net.train("create_pair_batch_random", batch_size, epochs, steps_per_epoch)

    # % Plot loss
    # Losses
    plotter = Plotter()
    plotter.plot_losses(net, history)

    # % Plot data MNIST
    # TRAIN DATA ############################################################################
    plotter.pca_plot_compare(X=dh.X_train, y=dh.y_train, net=net, suptitle='Train data')

    # % TEST DATA #############################################################################
    plotter.pca_plot_compare(X=dh.X_test, y=dh.y_test, net=net, suptitle='Test data')
    
    # % NEW DATA #############################################################################
    #plotter.pca_plot_compare(X=dh_newdata.X_test, y=dh_newdata.y_test, net=net, suptitle='Different data')

# %% NEW DATA
net.save_model()
net1 = SiameseNet(model_name='model4_alpha0.1_epochs1_batchSize200_steps1')

plotter.pca_plot_compare(X=dh.X_test, y=dh.y_test, net=net1, suptitle='hgfhfg data')
   