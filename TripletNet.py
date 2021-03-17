import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import numpy as np
from random import randint, sample

# TripletNet class.
class TripletNet:
    # Constructor for the TripletNet object.
    # model_handler - ModelHandler object.
    # data_handler  - DataHandler object.
    # alpha         - Loss margin for triplet loss.
    def __init__(self, model_handler, data_handler, alpha):
        self.data_handler       = data_handler
        self.input_feature_size = data_handler.n_features
        self.embedding_model    = model_handler.model
        self.embedding_size     = model_handler.embedding_size
        self.alpha              = alpha

        self.create_triplet_model()
    
    # Create triplet model.
    def create_triplet_model(self):
        # Define input tensors.
        input_anchor   = Input(shape=(self.input_feature_size,))
        input_positive = Input(shape=(self.input_feature_size,))
        input_negative = Input(shape=(self.input_feature_size,))
        
        # Define model.
        self.net = TripletModel(self.embedding_model)
        self.net([input_anchor, input_positive, input_negative])
        
        # Compile the model.
        self.net.compile(loss=self.triplet_loss, optimizer='adam')
            
    # Print model summary.
    def print_model(self):
        self.net.summary()
    
    # Data generator for model training.
    def data_generator(self):
        while True:
            x = self.create_a_batch_callback.xs
            y = self.create_a_batch_callback.ys
            yield x, y
    
    # Triplet loss.
    def triplet_loss(self, y_true, y_pred):
        anchor, positive, negative = y_pred[:,:self.embedding_size], y_pred[:,self.embedding_size:2*self.embedding_size], y_pred[:,2*self.embedding_size:]
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        return tf.maximum(positive_dist - negative_dist + self.alpha, 0.)
    
    # Train the model.
    def train(self, create_batch_function_name, number_of_samples, epochs, steps_per_epoch):
        if create_batch_function_name == "create_triplet_batch_random":
            self.create_a_batch_callback = TrainingCallback(self.data_handler, create_batch_function_name, batch_size=number_of_samples)
        else:
            self.create_a_batch_callback = TrainingCallback(self.data_handler, create_batch_function_name, samples_per_class=number_of_samples)
        
        # Create triplets for validation. TODO: See how big this should be.
        [x_anchors, x_positives, x_negatives], ys = self.data_handler.create_validation_triplets()
        validation_data = ([x_anchors, x_positives, x_negatives], ys)
        
        # Train the model.
        history = self.net.fit(
            self.data_generator(),
            steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=True,
            validation_data=validation_data, callbacks=[self.create_a_batch_callback])
        
        return history

# TripletModel class.
class TripletModel(Model):
    def __init__(self, embedding_model):
        super(TripletModel, self).__init__()
        self.embedding_model = embedding_model

    def call(self, inputs, training=False):
        embedded_anchor   = self.embedding_model(inputs[0]) # input_anchor
        embedded_positive = self.embedding_model(inputs[1]) # input_positive
        embedded_negative = self.embedding_model(inputs[2]) # input_negative
        output = concatenate([embedded_anchor, embedded_positive, embedded_negative], axis=1)

        return output

# TainingCallback class. Used for generating new batches on batch_end.
class TrainingCallback(Callback):
    # Constructor for the TrainingCallback object.
    # data_handler        - DataHandler object.
    # batch_function_name - Name of the batch function to use for selecting batch samples.
    # batch_size          - Optional parameter, should be provided if batch_function_name="create_triplet_batch_random"
    # samples_per_class   - Optional parameter, should be provided if batch_function_name="batch_hard" or "batch_all".
    def __init__(self, data_handler, batch_function_name, batch_size=0, samples_per_class=0):
        self.data_handler        = data_handler
        self.batch_function_name = batch_function_name
        self.batch_size          = batch_size
        if (batch_size == 0):
            self.batch_size = samples_per_class * data_handler.n_classes
        self.samples_per_class   = samples_per_class
        self.xs, self.ys         = self.create_triplet_batch_random()

    # Callback function that is called at each batch end. Here, a new batch is formed.
    def on_train_batch_end(self, batch, logs=None):
        if self.batch_function_name == "batch_hard":
            self.xs, self.ys = self.batch_hard()
        elif self.batch_function_name == "batch_all":
            self.xs, self.ys = self.batch_all()
        elif self.batch_function_name == "create_triplet_batch_random":
            self.xs, self.ys = self.create_triplet_batch_random()
        else:
            print("Invalid function name. Exiting program...")
            raise SystemExit(0)
    
    # Create a random batch of batch_size number of triplets.
    def create_triplet_batch_random(self):
        x_anchors = np.zeros((self.batch_size, self.data_handler.n_features))
        x_positives = np.zeros((self.batch_size, self.data_handler.n_features))
        x_negatives = np.zeros((self.batch_size, self.data_handler.n_features))
        ys = np.zeros(self.batch_size)

        for i in range(0, self.batch_size):
            # Choose anchor randomly.
            random_index = randint(0, self.data_handler.n_train - 1)
            x_anchor = self.data_handler.X_train[random_index]
            y = self.data_handler.y_train[random_index]

            # Based on the anchor, determine which samples are positive, 
            # and which ones are negative. Get their indices.
            positive_indices = np.squeeze(np.where(self.data_handler.y_train == y))
            negative_indices = np.squeeze(np.where(self.data_handler.y_train != y))

            # Choose a random positive. We assume that there is at least one positive.
            if positive_indices.ndim != 0:
                x_positive = self.data_handler.X_train[positive_indices[randint(0, len(positive_indices) - 1)]]
            else:
                x_positive = self.data_handler.X_train[positive_indices]

            # Choose a random negative.
            x_negative = self.data_handler.X_train[negative_indices[randint(0, len(negative_indices) - 1)]]

            # Declare i-th triplet.
            x_anchors[i]   = x_anchor
            x_positives[i] = x_positive
            x_negatives[i] = x_negative
            
        return [x_anchors, x_positives, x_negatives], ys
    
    # Choose a random set of samples and then form all possible triplets out of them.
    def batch_all(self):
        K = self.samples_per_class      # Samples per class.
        P = self.data_handler.n_classes # Number of classes.
        
        # Initialize batched samples arrays.
        batched_samples_X = np.zeros((P*K, self.data_handler.n_features))
        batched_samples_y = np.zeros(P*K)
        
        for i in range(0, P):
            # Choose K random samples from ith class.
            ith_class_indices = np.squeeze(np.where(self.data_handler.y_train == self.data_handler.class_labels[i]))
            random_indices    = sample(range(0, len(ith_class_indices)), K)

            batched_samples_X[i*K:(i+1)*K, :] = self.data_handler.X_train[random_indices]
            batched_samples_y[i*K:(i+1)*K]    = np.array(K * [i])
        
        # Initialize triplet arrays.
        total_batch_size = P*K*(P*K - K)*(K - 1)
        x_anchors   = np.zeros((total_batch_size, self.data_handler.n_features))
        x_positives = np.zeros((total_batch_size, self.data_handler.n_features))
        x_negatives = np.zeros((total_batch_size, self.data_handler.n_features))
        ys = np.zeros(total_batch_size)
        
        # Create all possible triplets from the selected batch samples.
        all_indices = np.arange(0, P*K, 1)
        samples_counter = 0
        for i in range(0, P*K):
            x_anchor = batched_samples_X[i]
            class_number = int(batched_samples_y[i])
            positive_indices = range(class_number*K, (class_number+1)*K)
            negative_indices = np.delete(all_indices, positive_indices)
            for j in positive_indices:
                if i != j: # positive != anchor
                    x_positive = batched_samples_X[j]
                    for k in negative_indices:
                        x_negative = batched_samples_X[k]
                        
                        x_anchors[samples_counter]   = x_anchor
                        x_positives[samples_counter] = x_positive
                        x_negatives[samples_counter] = x_negative
                        samples_counter += 1
                        
        return [x_anchors, x_positives, x_negatives], ys
    
    # Choose a random set of samples and then, for each sample as anchor, form
    # triplets by selecting the hardest negative and the hardest positive sample.
    def batch_hard(self):
        K = self.samples_per_class      # Number of samples per class.
        P = self.data_handler.n_classes # Number of classes.
        
        # Initialize batched samples arrays.
        batched_samples_X = np.zeros((P*K, self.data_handler.n_features))
        batched_samples_y = np.zeros(P*K)
        
        for i in range(0, P):
            # Choose k random samples from ith class.
            ith_class_indices = np.squeeze(np.where(self.data_handler.y_train == self.data_handler.class_labels[i]))
            random_indices    = sample(range(0, len(ith_class_indices)), K)

            batched_samples_X[i*K:(i+1)*K, :] = self.data_handler.X_train[random_indices]
            batched_samples_y[i*K:(i+1)*K]    = np.array(K * [i])
        
        # Predict output for the batched samples.
        preds = self.model.embedding_model.predict(batched_samples_X)
        
        # Initialize triplet arrays.
        total_batch_size = P*K
        x_anchors   = np.zeros((total_batch_size, self.data_handler.n_features))
        x_positives = np.zeros((total_batch_size, self.data_handler.n_features))
        x_negatives = np.zeros((total_batch_size, self.data_handler.n_features))
        ys = np.zeros(total_batch_size)
        
        # Create hard triplets from the selected batch samples.
        # i.e. select only the hardest positive and negative for each sample
        all_indices = np.arange(0, P*K, 1)
        for i in range(0, total_batch_size):
            # Declare anchor.
            x_anchor = batched_samples_X[i]
            x_embedded_anchor = preds[i]
            
            # See which class the anchor belongs to, and determine positive and negative indices.
            class_number = int(batched_samples_y[i])
            positive_indices = list(range(class_number*K, (class_number+1)*K))
            positive_indices.remove(i)
            negative_indices = np.delete(all_indices, positive_indices)
            
            # Calculate positive distances.
            positives = preds[positive_indices]
            embeddded_anchor_cloned = np.array([x_embedded_anchor,]*len(positive_indices))
            positive_dist = tf.reduce_mean(tf.square(embeddded_anchor_cloned - positives), axis=1)
            
            # Choose the hardest positive - the most far away sample.
            hardest_positive_ind = np.argmax(positive_dist)
            
            # Calculate negative distances.
            negatives = preds[negative_indices]
            embeddded_anchor_cloned = np.array([x_embedded_anchor,]*len(negative_indices))
            negative_dist = tf.reduce_mean(tf.square(embeddded_anchor_cloned - negatives), axis=1)
            
            # Choose the hardest negative - the closest sample.
            hardest_negative_ind = np.argmax(negative_dist)

            # Declare i-th triplet.
            x_anchors[i]   = x_anchor
            x_positives[i] = batched_samples_X[hardest_positive_ind]
            x_negatives[i] = batched_samples_X[hardest_negative_ind]

        return [x_anchors, x_positives, x_negatives], ys
