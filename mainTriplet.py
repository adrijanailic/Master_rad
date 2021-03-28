from TripletNet import TripletNet
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from plotters import Plotter

#%% MNIST dataset
dh = DataHandler("MNIST", [7,9])

# %% Define embedding model
mh = ModelHandler(model_number=4, embedding_size=200, input_feature_dim=dh.shape)

#%% Define triplet net
#alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
alphas = [0.1]
#mining_methods = ['batch_all', 'batch_hard', 'create_triplet_batch_random']
mining_methods = ['batch_hard']
#numbers_of_samples_per_class = [5, 10, 15, 20, 25]
numbers_of_samples_per_class = [15]
#epochs = [5, 10, 20, 30, 50, 100, 150, 200]
epochs = [20]
batch_sizes = [300]

for method in mining_methods:
    if method == 'batch_all' or method == 'batch_hard':
        batch_sizes_or_numbers_of_samples = numbers_of_samples_per_class
    else:
        batch_sizes_or_numbers_of_samples = batch_sizes
        
    for samples_count in batch_sizes_or_numbers_of_samples:
        for alpha in alphas:
            for epoch in epochs: 
                net = TripletNet(mh, dh, alpha)
                net.print_model()
                
                if method == 'batch_all':
                    K = samples_count # Samples per class.
                    P = dh.n_classes  # Number of classes.
                    total_batch_size = P*K*(P*K - K)*(K - 1)
                    steps_per_epoch = int(dh.n_train/total_batch_size)

                elif method == 'batch_hard':
                    K = samples_count # Samples per class.
                    P = dh.n_classes  # Number of classes.
                    total_batch_size = P*K
                    steps_per_epoch = 1#int(dh.n_train/total_batch_size)
                    epoch = 150

                else:
                    # A common practice is to set this value to number of samples/batch_size
                    # so that the model sees the training samples at most once per epoch.
                    steps_per_epoch = int(dh.n_train/samples_count)
                
                history = net.train(method, samples_count, epoch, steps_per_epoch)
                net.save_model()
                
                # % Plot loss
                # Losses
                plotter = Plotter()
                plotter.plot_losses(net, history)
            
                # % Plot data MNIST
                # TRAIN DATA ############################################################################
                plotter.pca_plot(X=dh.X_train, y=dh.y_train, net=net, suptitle='Train data')
                # TODO include title when plotting, add stuff like embedding size, model number?, batch size, ....
            
                # % TEST DATA #############################################################################
                plotter.pca_plot(X=dh.X_test, y=dh.y_test, net=net, suptitle='Test data')
                
                # % NEW DATA #############################################################################
                #plotter.set_plot_parameters(workspace)
                #plotter.pca_plot(X=dh_newdata.X_test, y=dh_newdata.y_test, net=net, suptitle='Different data')

# %%
#dhnew = DataHandler("MNIST", [7,9])
#plotter.scatter(dhnew.X_train, dhnew.y_train)
#net.save_model()
#net1 = TripletNet(model_name='model5_mining-batch_hard_alpha0.1_epochs10_numberOfSamplesPerClass2_steps1')

#plotter.pca_plot(X=dh.X_test, y=dh.y_test, net=net1, suptitle='hgfhfg data')
   