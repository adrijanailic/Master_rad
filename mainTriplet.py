from TripletNet import TripletNet
from DataHandler import DataHandler
from ModelHandler import ModelHandler
from plotters import Plotter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

plotter = Plotter()
#%% MNIST dataset
#dh = DataHandler("MNIST")

#%% fashionMNIST
dh = DataHandler("fashion_MNIST")

#%% Define triplet net
alphas = [0.1, 0.5, 1, 1.5, 2]

#mining_methods = ['batch_all', 'batch_hard', 'create_triplet_batch_random']
mining_methods = ['batch_hard']

#numbers_of_samples_per_class = [5, 10, 15, 20, 25]
numbers_of_samples_per_class = [5]
#epochs = [5, 10, 20, 30, 50, 100, 150, 200]
epochs = [50]
batch_sizes = [20]
knn_ks = [5, 10, 25, 50, 75, 100]
embedding_sizes = [5, 10, 50, 100]
steps_per_epoch = 10
accuracies_k = []


for method in mining_methods:
    if method == 'create_triplet_batch_random':
        batch_sizes_or_numbers_of_samples = batch_sizes
    else:
        batch_sizes_or_numbers_of_samples = numbers_of_samples_per_class

    for embedding_size in embedding_sizes:
        mh = ModelHandler(model_number=5, embedding_size=embedding_size, input_feature_dim=dh.shape)
        textfile = open('pics/triplet/embedding_size_' + str(embedding_size) + '.txt', 'w')
        
        for samples_count in batch_sizes_or_numbers_of_samples:
            for alpha in alphas:
                textfile.write('alpha = ' + str(alpha) + ': ')
                
                for epoch in epochs:
                    net = TripletNet(mh, dh, alpha)
                    net.print_model()
                    
                    if method == 'batch_all':
                        K = samples_count # Samples per class.
                        P = dh.n_classes  # Number of classes.
                        total_batch_size = P*K*(P*K - K)*(K - 1)
    
                    elif method == 'batch_hard':
                        K = samples_count # Samples per class.
                        P = dh.n_classes  # Number of classes.
                        total_batch_size = P*K
                        
                    elif method == 'batch_semihard':
                        K = samples_count # Samples per class.
                        P = dh.n_classes  # Number of classes.
                        total_batch_size = P*K*(K - 1)
    
                    
                    # Train model with defined parameters.
                    history = net.train(method, samples_count, epoch, steps_per_epoch)
                    #net.save_model()
                    
                    # % Plot losses.
                    plotter.plot_losses(net, history)
                
                    # % Plot data.
                    # TRAIN DATA ############################################################################
                    plotter.tsne_plot_compare(X=dh.X_train, y=dh.y_train, net=net, suptitle='Train data')
                    plotter.pca_plot_compare(X=dh.X_train, y=dh.y_train, net=net, suptitle='Train data')
                    
                    # % TEST DATA #############################################################################
                    plotter.tsne_plot_compare(X=dh.X_test, y=dh.y_test, net=net, suptitle='Test data')
                    plotter.pca_plot_compare(X=dh.X_test, y=dh.y_test, net=net, suptitle='Test data')
    
    
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
                    
                    for element in tmp:
                        textfile.write(str(element) + ", ")
                    
                    print(alpha)
                    print(accuracies_k)
                    
                textfile.write('\n')
        textfile.close()
