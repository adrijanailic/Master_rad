from SiameseNet import SiameseNet
from ModelHandler import ModelHandler
from DataHandler import DataHandler
from plotters import Plotter

# %% LFW dataset
# dh = DataHandler.DataHandler("LFW")
# X_train_sorted, y_train_sorted = DataHandler.sort_by_occurrence(X_train, y_train)
# X_test_sorted, y_test_sorted = DataHandler.sort_by_occurrence(X_test, y_test)

# %% MNIST dataset
dh = DataHandler("MNIST", classes_to_select=[2,3,5])

# %% Define embedding model
mh = ModelHandler(model_number=5, embedding_size=200, input_feature_dim=dh.shape)

# %% Define siamese net
net = SiameseNet(mh, dh, alpha=0.5)
net.print_model()
batch_size = 200
epochs = 2
steps_per_epoch = int(dh.n_train / batch_size)
history = net.train(dh.create_pair_batch_random, batch_size, epochs, steps_per_epoch)

# %% Plot loss
# Losses
plotter = Plotter()
plotter.plot_losses(history, epochs)

# %% Plot data MNIST
# TRAIN DATA ############################################################################
plotter.pca_plot(X=dh.X_train, y=dh.y_train, embedding_model=net.embedding_model)
# TODO include title when plotting, add stuff like embedding size, model number?, batch size, ....
#
# TEST DATA #############################################################################
plotter.pca_plot(X=dh.X_test, y=dh.y_test, embedding_model=net.embedding_model)

