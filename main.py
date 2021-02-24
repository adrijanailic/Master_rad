from TripletNet import TripletNet
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from plotters import PCAPlotter

#%% LFW dataset
#dh = DataHandler.DataHandler("LFW")
#X_train_sorted, y_train_sorted = DataHandler.sort_by_occurence(dh.X_train, dh.y_train)
#X_test_sorted, y_test_sorted = DataHandler.sort_by_occurence(dh.X_test, dh.y_test)

#%% MNIST dataset
dh = DataHandler("MNIST", classes_to_select=[0,4,5])

# %% Define embedding model
mh = ModelHandler(model_number=4, embedding_size=200, input_feature_dim=dh.shape)

#%% Define triplet net
net = TripletNet(mh, dh, alpha=0.3)
net.print_model()
batch_size = 300
epochs = 5 # 60
# A common practice is to set this value to number of samples/batch_size
# so that the model sees the training samples at most once per epoch.
steps_per_epoch = int(dh.n_train/batch_size) 
net.train(dh.create_triplet_batch_random, batch_size, epochs, steps_per_epoch)     

##%% Plot data LFW
## TRAIN DATA
#plotter = PCAPlotter(X=X_train_sorted[1:200,:], y=y_train_sorted[1:200], embedding_model=net.embedding_model)
#plotter.plot()
#
## TEST DATA
#plotter = PCAPlotter(X=X_test_sorted[1:300,:], y=y_test_sorted[1:300], embedding_model=net.embedding_model)
#plotter.plot()
#%% Plot data MNIST
# TRAIN DATA
plotter = PCAPlotter(X=dh.X_train, y=dh.y_train, embedding_model=net.embedding_model)
plotter.plot()

# TEST DATA
plotter = PCAPlotter(X=dh.X_test, y=dh.y_test, embedding_model=net.embedding_model)
plotter.plot()