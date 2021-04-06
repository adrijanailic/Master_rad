from TripletNet import TripletNet
from DataHandler import DataHandler
from plotters import Plotter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#%% MNIST dataset
dh = DataHandler("MNIST")
#%%
knn_ks = [1, 5, 10, 50, 100, 500]
net = TripletNet(model_name='model4_mining-create_triplet_batch_random_alpha0.15_epochs20_batchSize300_steps10')
X_embedded_train = net.embedding_model.predict(dh.X_train)
X_embedded_test  = net.embedding_model.predict(dh.X_test)
accuracies_k = []
for k in knn_ks:
    print("Start training...")
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_embedded_train, dh.y_train)
    y_pred = knn_classifier.predict(X_embedded_test)
    accuracies_k.append(accuracy_score(dh.y_test, y_pred))
    print(accuracies_k)
                
#%%
plotter = Plotter()
plotter.tsne_plot_compare(X=dh.X_train, y=dh.y_train, net=net, suptitle='Train data')
            
# % TEST DATA #############################################################################
plotter.tsne_plot_compare(X=dh.X_test, y=dh.y_test, net=net, suptitle='Test data')
                