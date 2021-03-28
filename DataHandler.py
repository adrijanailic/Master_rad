import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# DataHandler class.
class DataHandler:
    # Constructor.
    # data_to_load      - Name of the dataset to load.
    # classes_to_select - Classes to select from the dataset.
    def __init__(self, data_to_load, classes_to_select=[]):
        self.dataset_name = data_to_load
            
        if data_to_load == "MNIST":
            # The MNIST database of handwritten digits. It contains:
            #    -training set of 60,000 examples 
            #    -test set of 10,000 examples
            # Labels are 0-9.
            self.n_classes = 10
            self.class_labels = np.arange(0, 10, 1)

            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
            self.shape = (self.X_train.shape[1], self.X_train.shape[2], 1)  # Original shape, before reshaping.

            # X_train.shape: (60000, 28, 28) --> (60000, 784)
            self.X_train = np.reshape(self.X_train,
                                     (self.X_train.shape[0], self.X_train.shape[1] * self.X_train.shape[2])) / 255.
            self.X_test  = np.reshape(self.X_test,
                                     (self.X_test.shape[0], self.X_test.shape[1] * self.X_test.shape[2])) / 255.
            
            # Create an extra validation set.
            self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(self.X_train, self.y_train, 
                                                                                            test_size=0.15, random_state=42)

        elif data_to_load == "fashion_MNIST":
            # The fashion_MNIST database of clothing pictures. It contains:
            #    -training set of 60,000 examples 
            #    -test set of 10,000 examples
            # Labels are 0-9.
            self.n_classes = 10
            self.class_labels = np.arange(0, 10, 1)

            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
            self.shape = (self.X_train.shape[1], self.X_train.shape[2], 1)  # Original shape, before reshaping.

            # X_train.shape: (60000, 28, 28) --> (60000, 784)
            self.X_train = np.reshape(self.X_train,
                                     (self.X_train.shape[0], self.X_train.shape[1] * self.X_train.shape[2])) / 255.
            self.X_test  = np.reshape(self.X_test,
                                     (self.X_test.shape[0], self.X_test.shape[1] * self.X_test.shape[2])) / 255.
            
            # Create an extra validation set.
            self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(self.X_train, self.y_train, 
                                                                                            test_size=0.15, random_state=42)
        # Select only certain classes from the dataset.
        if classes_to_select:
            self.select_classes(classes_to_select)
            self.n_classes = len(classes_to_select)
            self.class_labels = np.array(classes_to_select)

        self.n_features = self.X_train.shape[1]
        self.n_train    = self.X_train.shape[0]
        self.n_test     = self.X_test.shape[0]
        self.n_validate = self.X_validate.shape[0]


    # Select only certain classes from the dataset.
    # classes_to_select - List of classes to select.
    def select_classes(self, classes_to_select):
        train_mask   = np.isin(self.y_train, classes_to_select)
        self.X_train = self.X_train[train_mask]
        self.y_train = self.y_train[train_mask]

        test_mask   = np.isin(self.y_test, classes_to_select)
        self.X_test = self.X_test[test_mask]
        self.y_test = self.y_test[test_mask]
        
        validate_mask   = np.isin(self.y_validate, classes_to_select)
        self.X_validate = self.X_validate[validate_mask]
        self.y_validate = self.y_validate[validate_mask]