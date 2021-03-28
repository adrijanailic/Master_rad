import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from random import randint, random
from ModelHandler import ModelHandler
from DataHandler  import DataHandler

# SiameseNet class.
class SiameseNet:
    # Constructor for the SiameseNet object.
    # Object can be constructed either from scratch, in which case model_handler,
    # data_handler and alpha must be provided, or, alternatively, a premade object
    # can be loaded by specifying the model_name.
    # model_handler - ModelHandler object.
    # data_handler  - DataHandler object.
    # alpha         - Loss margin for contrastive loss.
    def __init__(self, model_handler=0, data_handler=0, alpha=0, model_name=0):
        if model_name == 0: 
            self.data_handler       = data_handler
            self.input_feature_size = data_handler.n_features
            self.model_handler      = model_handler
            self.embedding_model    = model_handler.model
            self.embedding_size     = model_handler.embedding_size
            self.alpha              = alpha
            self.create_siamese_net()
        else:
            self.load_model(model_name)

    # Create siamese net.
    def create_siamese_net(self):
        # Define input tensors.
        input_1 = Input(shape=(self.input_feature_size,))
        input_2 = Input(shape=(self.input_feature_size,))

        # Define model.
        self.net = SiameseModel(self.embedding_model)
        self.net([input_1, input_2])

        # Compile the model.
        self.net.compile(loss=self.contrastive_loss, optimizer='adam')                

    # Print model summary.
    def print_model(self):
        self.net.summary()
        
    # Save model weights and attributes.
    # model_name - The name of the file where to store the model weights and
    #              the name of the .pkl file to store the model attributes.
    def save_model(self, model_name=''):
        model_attributes = {
            "net": "siamese",
            "dataset_name": self.data_handler.dataset_name,
            "classes": self.data_handler.class_labels.tolist(),
            "alpha": self.alpha,
            "model_number": self.model_handler.model_number,
            "embedding_size": self.model_handler.embedding_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "mining_method": self.mining_method
            }
        
        if model_name == '':
            model_name = 'model' + str(model_attributes["model_number"]) + '_alpha' \
                + str(model_attributes["alpha"]) + '_epochs' + str(model_attributes["epochs"]) \
                + '_batchSize' + str(model_attributes["batch_size"]) \
                + '_steps' + str(model_attributes["steps_per_epoch"])
    
        with open('./models/contrastive/' + model_name + '.pkl', 'wb') as file:
            pickle.dump(model_attributes, file)
        
        self.net.save_weights('./models/contrastive/' + model_name)
    
    # Load model.
    # model_name - The name of the file where the model weights are stored and
    #              the name of the .pkl file where the model attributes are stored.
    def load_model(self, model_name):
        with open('./models/contrastive/' + model_name + '.pkl', 'rb') as file:
            model_attributes = pickle.load(file)
        
        self.data_handler       = DataHandler(model_attributes['dataset_name'], model_attributes['classes'])
        self.input_feature_size = self.data_handler.n_features
        self.model_handler      = ModelHandler(model_attributes['model_number'], model_attributes['embedding_size'], input_feature_dim=self.data_handler.shape)
        self.embedding_model    = self.model_handler.model
        self.embedding_size     = self.model_handler.embedding_size
        self.alpha              = model_attributes['alpha']
        self.batch_size         = model_attributes['batch_size']
        self.epochs             = model_attributes['epochs']
        self.steps_per_epoch    = model_attributes['steps_per_epoch']
        self.mining_method      = model_attributes['mining_method']
        
        self.create_siamese_net()

        self.net.load_weights('./models/contrastive/' + model_name)

    # Data generator for model training.
    def data_generator(self):
        while True:
            x = self.create_a_batch_callback.xs
            y = self.create_a_batch_callback.ys
            yield x, y

    # TODO: check if needed!
    @staticmethod
    def euclidian_distance(vectors):
        (positive, negative) = vectors
        return tf.reduce_mean(tf.square(positive - negative), axis=1)

    # Contrastive loss.
    # y_true - Set of correct labels. Here, y_true is a vector of similarities between the two, paired vectors.
    #          Value is 0 for similar points and 1 for dissimilar points.
    # y_pred - Set of predicted values - embedded points.
    def contrastive_loss(self, y_true, y_pred):
        positive, negative = y_pred[:, :self.embedding_size], y_pred[:, self.embedding_size:]
        dist = tf.reduce_mean(tf.square(positive - negative), axis=1)
        return (1 - y_true) * dist + y_true * tf.maximum(0., self.alpha - dist)
    
    # Train the model.
    # create_batch_function_name - Name of the mining method.
    #                              It can be one of the following: "create_pair_batch_random".
    # batch_size                 - Batch size.
    # epochs                     - Number of epochs.
    # steps_per_epoch            - Steps per epoch.
    def train(self, create_batch_function_name, batch_size, epochs, steps_per_epoch):
        # Save parameters.
        self.mining_method   = create_batch_function_name
        self.batch_size      = batch_size
        self.epochs          = epochs
        self.steps_per_epoch = steps_per_epoch
            
        # Create a callback object.
        self.create_a_batch_callback = TrainingCallback(self.data_handler, create_batch_function_name, batch_size=batch_size)
        
        # Create validation pairs.
        # TODO: How many validation pairs to use? For now I use default value of 1000.
        [x_1s, x_2s], ys = self.create_validation_pairs()
        validation_data = ([x_1s, x_2s], ys)
        
        # Train the model.
        history = self.net.fit(
            self.data_generator(),
            steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=True,
            validation_data=validation_data, callbacks=[self.create_a_batch_callback])
        
        return history
    
    # Create number_of_pairs validation pairs.
    # number_of_pairs - Number of pairs to generate.
    def create_validation_pairs(self, number_of_pairs=1000):
        # Initialize arrays.
        x_1s = np.zeros((number_of_pairs, self.data_handler.n_features))
        x_2s = np.zeros((number_of_pairs, self.data_handler.n_features))
        # Similarity array: 0 - data from the same class, 1 - different class.
        ys = np.zeros(number_of_pairs)

        for i in range(0, number_of_pairs):
            # Choose first element x_1 randomly.
            random_index = randint(0, self.data_handler.n_validate - 1)
            x_1 = self.data_handler.X_validate[random_index]
            y_1 = self.data_handler.y_validate[random_index]

            # Choose second element x_2 randomly.
            random_index = randint(0, self.data_handler.n_validate - 1)
            x_2 = self.data_handler.X_validate[random_index]
            y_2 = self.data_handler.y_validate[random_index]

            if y_1 == y_2:
                ys[i] = 0
            else:
                ys[i] = 1

            # Declare i-th pair.
            x_1s[i] = x_1
            x_2s[i] = x_2
                
        return [x_1s, x_2s], ys
    
# SiameseModel class, derived from tensorflow Model class.
class SiameseModel(Model):
    # Constructor for the SiameseModel class.
    # embedding_model - Base model with which to construct siamese net.
    def __init__(self, embedding_model):
        super(SiameseModel, self).__init__()
        self.embedding_model = embedding_model

    # Construct Siamese net.
    def call(self, inputs, training=False):
        embedded_1 = self.embedding_model(inputs[0]) # input_1
        embedded_2 = self.embedding_model(inputs[1]) # input_2
        # distance = Lambda(self.euclidean_distance)([embedded_1, embedded_2])
        # output = Dense(1, activation="sigmoid")(distance)
        output = concatenate([embedded_1, embedded_2], axis=1)

        return output

# TainingCallback class. Used for generating new batches on batch_end.
class TrainingCallback(Callback):
    # Constructor for the TrainingCallback object.
    # data_handler         - DataHandler object.
    # batch_function_name  - Name of the batch function to use for selecting batch samples.
    # batch_size           - Batch size.
    # same_class_frequency - Optional parameter. Frequency with which pairs of the same class are generated.
    #                        If none is specified. The frequency will be 1/number of classes.
    #                        This parameter should take values in the range of [0, 1].
    def __init__(self, data_handler, batch_function_name, batch_size, same_class_frequency=-1):
        self.data_handler        = data_handler
        self.batch_function_name = batch_function_name
        self.batch_size          = batch_size
        if same_class_frequency == -1:
            self.same_class_frequency = 1/self.data_handler.n_classes
        else:
            self.same_class_frequency = same_class_frequency
            
        self.xs, self.ys = self.create_pair_batch_random(self.same_class_frequency)

    # Callback function that is called at each batch end. Here, a new batch is formed.
    def on_train_batch_end(self, batch, logs=None):
        if self.batch_function_name == "create_pair_batch_random":
            self.xs, self.ys = self.create_pair_batch_random(self.same_class_frequency)
        else:
            print("Invalid function name. Exiting program...")
            raise SystemExit(0)
    
    # Create a batch of batch_size number of pairs. Batches are created randomly.
    # same_class_frequency - Frequency with which pairs of the same class are generated.
    def create_pair_batch_random(self, same_class_frequency=-1):        
        x_1s = np.zeros((self.batch_size, self.data_handler.n_features))
        x_2s = np.zeros((self.batch_size, self.data_handler.n_features))
        # Similarity array: 0 - data from the same class, 1 - different class.
        ys = np.zeros(self.batch_size)

        for i in range(0, self.batch_size):
            # Choose first element x_1 randomly.
            random_index = randint(0, self.data_handler.n_train - 1)
            x_1 = self.data_handler.X_train[random_index]
            y_1 = self.data_handler.y_train[random_index]

            # Find indices of similar and dissimilar elements.
            similar_indices = np.squeeze(np.where(self.data_handler.y_train == y_1))
            dissimilar_indices = np.squeeze(np.where(self.data_handler.y_train != y_1))

            # Most of the data is going to be dissimilar.
            # With choose_probability we want to give some advantage to the similar data as well.
            # TODO: This parameter has some powerful effects on the results. Document it!
            choose_probability = random()

            if choose_probability < same_class_frequency:
                # Choose a random similar example.
                ys[i] = 0
                # We assume that there is at least one similar example.
                if similar_indices.ndim != 0:
                    random_index = randint(0, len(similar_indices) - 1)
                    x_2 = self.data_handler.X_train[similar_indices[random_index]]
                else:
                    x_2 = self.data_handler.X_train[similar_indices]
            else:
                # Choose a random dissimilar example.
                ys[i] = 1
                random_index = randint(0, len(dissimilar_indices) - 1)
                x_2 = self.data_handler.X_train[dissimilar_indices[random_index]]

            # Declare i-th pair.
            x_1s[i] = x_1
            x_2s[i] = x_2
                
        return [x_1s, x_2s], ys
