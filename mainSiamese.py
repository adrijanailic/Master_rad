from SiameseNet import SiameseNet
from ModelHandler import ModelHandler
from DataHandler import DataHandler
from plotters import Plotter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

plotter = Plotter()

# %% MNIST dataset
dh = DataHandler("fashion_MNIST", classes_to_select=[5, 7, 9])
#dh = DataHandler("MNIST", classes_to_select=[7,8,9])

# %% Define siamese net
#alphas = [0.1, 0.5, 1, 1.5, 2]
alphas = [1]
epochs = [20]
batch_sizes = [300]
knn_ks = [5, 10, 25, 50, 75, 100]
#knn_ks = [1]
#embedding_sizes = [5, 10, 50, 100]
embedding_sizes = [100]


accuracies_k = []
for embedding_size in embedding_sizes:
    textfile = open('pics/contrastive/embedding_size_' + str(embedding_size) + '.txt', 'w')
    mh = ModelHandler(model_number=5, embedding_size=embedding_size, input_feature_dim=dh.shape)
    for alpha in alphas:
        textfile.write('alpha = ' + str(alpha) + ': ')
        for batch_size in batch_sizes:
            for epoch in epochs:
                net = SiameseNet(mh, dh, alpha)
                net.print_model()
                
                steps_per_epoch = 10 #int(dh.n_train / batch_size)
                history = net.train("create_pair_batch_random", batch_size, epoch, steps_per_epoch)
    
                # % Plot loss
                # Losses
                plotter.plot_losses(net, history)
    
                # % Plot data MNIST
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
        textfile.write('\n')
    textfile.close()

print(accuracies_k)
