import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
import numpy as np


class TripletNet:
    def __init__(self, model_handler, data_handler, alpha):
        self.input_feature_size = data_handler.n_features
        self.embedding_model = model_handler.model
        self.embedding_size = model_handler.embedding_size
        self.alpha = alpha                 # loss margin

        self.create_triplet_model()
        
    def create_triplet_model(self):
        # Define input tensors.
        input_anchor   = Input(shape=(self.input_feature_size,))
        input_positive = Input(shape=(self.input_feature_size,))
        input_negative = Input(shape=(self.input_feature_size,))
        
        # Get embedded outputs.
        embedded_anchor   = self.embedding_model(input_anchor)
        embedded_positive = self.embedding_model(input_positive)
        embedded_negative = self.embedding_model(input_negative)
        
        # Concatenate outputs. - TODO can be modified in the future.
        output = concatenate([embedded_anchor, embedded_positive, embedded_negative], axis=1)
        
        # Put everything together.
        self.net = Model([input_anchor, input_positive, input_negative], output)
        
        # Compile the model.
        self.net.compile(loss=self.triplet_loss, optimizer='adam')
            
    def print_model(self):
        self.net.summary()
        
    def data_generator(self, create_batch_function, batch_size=256):
        while True:
            x = create_batch_function(batch_size)
            y = np.zeros((batch_size, 3*self.embedding_size))
            yield x, y
        
    def triplet_loss(self, y_true, y_pred):
        anchor, positive, negative = y_pred[:,:self.embedding_size], y_pred[:,self.embedding_size:2*self.embedding_size], y_pred[:,2*self.embedding_size:]
        positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
        negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
        return tf.maximum(positive_dist - negative_dist + self.alpha, 0.)
    
    def train(self, create_batch_function, batch_size, epochs, steps_per_epoch):     
        _ = self.net.fit(
            self.data_generator(create_batch_function, batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs, verbose=True)