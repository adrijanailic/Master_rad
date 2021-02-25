import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import tensorflow as tf


class DataHandler:
    # Constructor.
    def __init__(self, data_to_load, classes_to_select=[]):
        if data_to_load == "LFW":
            # Download data and store as numpy arrays.
            lfw_people = fetch_lfw_people(min_faces_per_person=2)
            self.shape = (lfw_people.images.shape[1], lfw_people.images.shape[2], 1)  # (62, 47)

            X = lfw_people.data    # flattened images
            y = lfw_people.target  # person IDs
            # TODO In few shot learning scenario, we need to test on faces the model has never seen before,
            # TODO and not like this!
            # Split into training and testing sets.
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.X_test, self.X_validate, self.y_test, self.y_validate = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=42)

        elif data_to_load == "MNIST":
            # The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, 
            # and a test set of 10,000 examples. Labels are 0-9.

            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
            self.shape = (self.X_train.shape[1], self.X_train.shape[2], 1)  # Original shape, before reshaping.

            # X_train.shape: (60000, 28, 28) --> (60000, 784)
            self.X_train = np.reshape(self.X_train,
                                      (self.X_train.shape[0], self.X_train.shape[1] * self.X_train.shape[2])) / 255.
            self.X_test = np.reshape(self.X_test,
                                     (self.X_test.shape[0], self.X_test.shape[1] * self.X_test.shape[2])) / 255.
            
            # Create an extra validation set.
            self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(self.X_train, self.y_train, 
                                                                                            test_size=0.15, random_state=42)

        elif data_to_load == "fashion_MNIST":
            # The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, 
            # and a test set of 10,000 examples. Labels are 0-9.

            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
            self.shape = (self.X_train.shape[1], self.X_train.shape[2], 1)  # Original shape, before reshaping.

            # X_train.shape: (60000, 28, 28) --> (60000, 784)
            self.X_train = np.reshape(self.X_train,
                                      (self.X_train.shape[0], self.X_train.shape[1] * self.X_train.shape[2])) / 255.
            self.X_test = np.reshape(self.X_test,
                                     (self.X_test.shape[0], self.X_test.shape[1] * self.X_test.shape[2])) / 255.
            
            # Create an extra validation set.
            self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(self.X_train, self.y_train, 
                                                                                            test_size=0.15, random_state=42)
        # Select only certain classes from the dataset.
        if classes_to_select:
            self.select_classes(classes_to_select)

        self.n_features = self.X_train.shape[1]
        self.n_train = self.X_train.shape[0]
        self.n_test = self.X_test.shape[0]
        self.n_validate = self.X_validate.shape[0]

    # Select only certain classes from the dataset.
    def select_classes(self, classes_to_select):
        train_mask = np.isin(self.y_train, classes_to_select)  # classes_to_select je lista klasi npr [2,8]
        self.X_train = self.X_train[train_mask]                # zasad je classes to select za samo 2 klase.. np.array dole
        self.y_train = self.y_train[train_mask]

        test_mask = np.isin(self.y_test, classes_to_select)
        self.X_test = self.X_test[test_mask]
        self.y_test = self.y_test[test_mask]
        
        validate_mask = np.isin(self.y_validate, classes_to_select)
        self.X_validate = self.X_validate[validate_mask]
        self.y_validate = self.y_validate[validate_mask]

    # Sort data by occurrence. The most frequent data goes first.
    @staticmethod
    def sort_data_by_occurrence(X, y):
        unique, counts = np.unique(y, return_counts=True)
        occ = dict(zip(unique, counts))
        # Array, where each element represents number of occurrences of according element of y_test.
        occurrences = [occ.get(elem) for elem in y]

        indices = np.arange(len(y))
        zipped_lists = zip(occurrences, indices)
        sorted_pairs = sorted(zipped_lists, reverse=True)
        tuples = zip(*sorted_pairs)
        occR, indicesR = [list(tuple) for tuple in tuples]

        # indicesR je nacin na koji zelimo da sortiramo X_test i y_test
        X_sorted = X[indicesR]
        y_sorted = y[indicesR]

        return X_sorted, y_sorted

    # Create a batch of batch_size number of triplets. Batches are created randomly.
    def create_triplet_batch_random(self, batch_size=256):
        x_anchors = np.zeros((batch_size, self.n_features))
        x_positives = np.zeros((batch_size, self.n_features))
        x_negatives = np.zeros((batch_size, self.n_features))

        for i in range(0, batch_size):
            # Choose anchor randomly.
            random_index = random.randint(0, self.n_train - 1)
            x_anchor = self.X_train[random_index]
            y = self.y_train[random_index]

            # Based on the anchor, determine which samples are positive, 
            # and which ones are negative. Get their indices.
            positive_indices = np.squeeze(np.where(self.y_train == y))
            negative_indices = np.squeeze(np.where(self.y_train != y))

            # Choose a random positive. We assume that there is at least one positive.
            if positive_indices.ndim != 0:
                x_positive = self.X_train[positive_indices[random.randint(0, len(positive_indices) - 1)]]
            else:
                x_positive = self.X_train[positive_indices]

            # Choose a random negative.
            x_negative = self.X_train[negative_indices[random.randint(0, len(negative_indices) - 1)]]

            # Declare i-th triplet.
            x_anchors[i] = x_anchor
            x_positives[i] = x_positive
            x_negatives[i] = x_negative

        return [x_anchors, x_positives, x_negatives]

    # Create a batch of batch_size number of pairs. Batches are created randomly.
    def create_pair_batch_random(self, batch_size=256):
        x_1s = np.zeros((batch_size, self.n_features))
        x_2s = np.zeros((batch_size, self.n_features))
        ys = np.zeros(batch_size)  # Similarity array: 1 - data from the same class, 0 - different class.

        for i in range(0, batch_size):
            # Choose first element x_1 randomly.
            random_index = random.randint(0, self.n_train - 1)
            x_1 = self.X_train[random_index]
            y_1 = self.y_train[random_index]

            # Find indices of similar and dissimilar elements.
            similar_indices = np.squeeze(np.where(self.y_train == y_1))
            dissimilar_indices = np.squeeze(np.where(self.y_train != y_1))

            # Most of the data is going to be dissimilar.
            # With choose_probability we want to give some advantage to the positives as well.
            # This parameter has some damn powerful effect on the results.. Document it!!
            choose_probability = random.randint(1, 10)

            if choose_probability < 1:
                # Choose a random similar example.
                ys[i] = 0
                # We assume that there is at least one similar example.
                if similar_indices.ndim != 0:
                    random_index = random.randint(0, len(similar_indices) - 1)
                    x_2 = self.X_train[similar_indices[random_index]]
                else:
                    x_2 = self.X_train[similar_indices]
            else:
                # Choose a random dissimilar example.
                ys[i] = 1
                random_index = random.randint(0, len(dissimilar_indices) - 1)
                x_2 = self.X_train[dissimilar_indices[random_index]]

            # Declare i-th pair.
            x_1s[i] = x_1
            x_2s[i] = x_2
                
        return [x_1s, x_2s], ys
    
    # Create validation pairs. For now, I create only n_validate pairs, because otherwise we would
    # have too many of them.
    def create_validation_pairs(self):
        x_1s = np.zeros((self.n_validate, self.n_features))
        x_2s = np.zeros((self.n_validate, self.n_features))
        ys = np.zeros(self.n_validate)  # Similarity array: 0 - data from the same class, 1 - different class.

        for i in range(0, self.n_validate):
            # Choose first element x_1 randomly.
            random_index = random.randint(0, self.n_validate - 1)
            x_1 = self.X_validate[random_index]
            y_1 = self.y_validate[random_index]

            random_index = random.randint(0, self.n_validate - 1)
            x_2 = self.X_validate[random_index]
            y_2 = self.y_validate[random_index]

            if y_1 == y_2:
                ys[i] = 0
            else:
                ys[i] = 1

            # Declare i-th pair.
            x_1s[i] = x_1
            x_2s[i] = x_2
                
        return [x_1s, x_2s], ys
    
    # Create validation triplets. For now, I create only n_validate triplets, because otherwise we would
    # have too many of them.
    def create_validation_triplets(self):
        x_anchors = np.zeros((self.n_validate, self.n_features))
        x_positives = np.zeros((self.n_validate, self.n_features))
        x_negatives = np.zeros((self.n_validate, self.n_features))
        ys = np.zeros(self.n_validate)

        for i in range(0, self.n_validate):
            # Choose anchor randomly.
            random_index = random.randint(0, self.n_validate - 1)
            x_anchor = self.X_validate[random_index]
            y = self.y_validate[random_index]

            # Based on the anchor, determine which samples are positive, 
            # and which ones are negative. Get their indices.
            positive_indices = np.squeeze(np.where(self.y_validate == y))
            negative_indices = np.squeeze(np.where(self.y_validate != y))

            # Choose a random positive. We assume that there is at least one positive.
            if positive_indices.ndim != 0:
                x_positive = self.X_validate[positive_indices[random.randint(0, len(positive_indices) - 1)]]
            else:
                x_positive = self.X_validate[positive_indices]

            # Choose a random negative.
            x_negative = self.X_validate[negative_indices[random.randint(0, len(negative_indices) - 1)]]

            # Declare i-th triplet.
            x_anchors[i] = x_anchor
            x_positives[i] = x_positive
            x_negatives[i] = x_negative

        return [x_anchors, x_positives, x_negatives], ys
