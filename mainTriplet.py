from TripletNet import TripletNet
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from plotters import Plotter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#+ TODO tf_mean/tf_sum reduce...

plotter = Plotter()
#%% MNIST dataset
#dh = DataHandler("MNIST")
#plotter.tsne_plot(dh.X_train, dh.y_train, title='MNIST dataset', figname='mnist_dataset_tsne')

#%% fashionMNIST
dh = DataHandler("MNIST", [4, 9, 5])
#plotter.tsne_plot(dh_fashion.X_train, dh_fashion.y_train, title='fashion_MNIST dataset', figname='fashion_mnist_dataset_tsne')

# %% Define embedding model
mh = ModelHandler(model_number=5, embedding_size=50, input_feature_dim=dh.shape)

#%% Define triplet net
#alphas = [1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]
alphas = [4]
#mining_methods = ['batch_all', 'batch_hard', 'create_triplet_batch_random']
mining_methods = ['batch_hard']
# model collapse explained:
# https://stats.stackexchange.com/questions/475655/in-training-a-triplet-network-i-first-have-a-solid-drop-in-loss-but-eventually
#numbers_of_samples_per_class = [5, 10, 15, 20, 25]
numbers_of_samples_per_class = [100]
#epochs = [5, 10, 20, 30, 50, 100, 150, 200]
epochs = [10]
batch_sizes = [300]
knn_ks = [5, 10, 25, 50, 75, 100]
#knn_ks = [1]

accuracies_k = []


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
                    steps_per_epoch = 10 #int(dh.n_train/total_batch_size)

                elif method == 'batch_hard':
                    K = samples_count # Samples per class.
                    P = dh.n_classes  # Number of classes.
                    total_batch_size = P*K
                    steps_per_epoch = 10#int(dh.n_train/total_batch_size)
                    #♦epoch = 10

                else:
                    # A common practice is to set this value to number of samples/batch_size
                    # so that the model sees the training samples at most once per epoch.
                    steps_per_epoch = 10#int(dh.n_train/samples_count)
                
                # Train model with defined parameters.
                history = net.train(method, samples_count, epoch, steps_per_epoch)
                #net.save_model()
                
                # % Plot loss
                # Losses
                plotter.plot_losses(net, history)
            
                # % Plot data MNIST
                # TRAIN DATA ############################################################################
                plotter.tsne_plot_compare(X=dh.X_train, y=dh.y_train, net=net, suptitle='Train data')
                # TODO include title when plotting, add stuff like embedding size, model number?, batch size, ....
            
                # % TEST DATA #############################################################################
                plotter.tsne_plot_compare(X=dh.X_test, y=dh.y_test, net=net, suptitle='Test data')
                
                # % NEW DATA #############################################################################
                #plotter.pca_plot(X=dh_newdata.X_test, y=dh_newdata.y_test, net=net, suptitle='Different data')

                #% Train kNN classifier with embedded data.
                X_embedded_train = net.embedding_model.predict(dh.X_train)
                X_embedded_test  = net.embedding_model.predict(dh.X_test)
                tmp = []
                for k in knn_ks:
                    knn_classifier = KNeighborsClassifier(n_neighbors=k)
                    knn_classifier.fit(X_embedded_train, dh.y_train)
                    y_pred = knn_classifier.predict(X_embedded_test)
                    tmp.append(accuracy_score(dh.y_test, y_pred))
                accuracies_k.append(tmp)
                
                print(alpha)
                print(accuracies_k)
# %%
#dhnew = DataHandler("MNIST", [7,9])
#plotter.scatter(dhnew.X_train, dhnew.y_train)
#net.save_model()
#net1 = TripletNet(model_name='model4_mining-create_triplet_batch_random_alpha0.1_epochs3_batchSize300_steps170')

plotter.pca_plot_compare(X=dh.X_test, y=dh.y_test, net=net, suptitle='hgfhfg data')
#plotter = Plotter()
#plotter.pca_plot_compare(X=dh.X_train, y=dh.y_train, net=net1, suptitle='Train data')
#plotter.tsne_plot_compare(X=dh.X_train, y=dh.y_train, net=net, suptitle='Train data')
#plotter.plot_losses(net, history)


   