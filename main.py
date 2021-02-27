from TripletNet import TripletNet
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from plotters import Plotter

#%% LFW dataset
#dh = DataHandler.DataHandler("LFW")
#X_train_sorted, y_train_sorted = DataHandler.sort_by_occurence(dh.X_train, dh.y_train)
#X_test_sorted, y_test_sorted = DataHandler.sort_by_occurence(dh.X_test, dh.y_test)

#%% MNIST dataset
dh = DataHandler("MNIST", [0,1])

# %% Define embedding model
mh = ModelHandler(model_number=5, embedding_size=200, input_feature_dim=dh.shape)

#%% Define triplet net
net = TripletNet(mh, dh, alpha=0.2)
net.print_model()
batch_size = 200
epochs = 5 # 60
# A common practice is to set this value to number of samples/batch_size
# so that the model sees the training samples at most once per epoch.
steps_per_epoch = int(dh.n_train/batch_size) 
history = net.train(dh.create_triplet_batch_random, batch_size, epochs, steps_per_epoch)

#%% Plot losses
plotter = Plotter()
plotter.plot_losses(history, epochs)

#%% Plot data LFW
## TRAIN DATA
#plotter.pca_plot(X=X_train_sorted[1:200,:], y=y_train_sorted[1:200], embedding_model=net.embedding_model)
#
## TEST DATA
#plotter.pca_plot(X=X_test_sorted[1:300,:], y=y_test_sorted[1:300], embedding_model=net.embedding_model)
#%% Plot data MNIST
#dh = DataHandler("MNIST", classes_to_select=[7,9])

# TRAIN DATA
plotter.pca_plot(X=dh.X_train, y=dh.y_train, embedding_model=net.embedding_model)

# TEST DATA
plotter.pca_plot(X=dh.X_test, y=dh.y_test, embedding_model=net.embedding_model)