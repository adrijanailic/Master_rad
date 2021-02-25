import tensorflow as tf
#from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model


class SiameseNet:

    def __init__(self, model_handler, data_handler, alpha):
        self.data_handler = data_handler
        self.input_feature_size = data_handler.n_features
        self.embedding_model = model_handler.model
        self.embedding_size = model_handler.embedding_size
        self.alpha = alpha                 # loss margin

        self.create_siamese_net()

    def create_siamese_net(self):
        # Define input tensors.
        input_positive = Input(shape=(self.input_feature_size,))
        input_negative = Input(shape=(self.input_feature_size,))

        # Get embedded outputs.
        embedded_positive = self.embedding_model(input_positive)
        embedded_negative = self.embedding_model(input_negative)

        # distance = Lambda(self.euclidean_distance)([embedded_positive, embedded_negative])
        # output = Dense(1, activation="sigmoid")(distance)

        # Concatenate outputs. - TODO can be modified in the future.
        output = concatenate([embedded_positive, embedded_negative], axis=1)

        # Put everything together.
        self.net = Model([input_positive, input_negative], output)

        # Compile the model.
        self.net.compile(loss=self.contrastive_loss, optimizer='adam')                

    def print_model(self):
        self.net.summary()

    def data_generator(self, create_batch_function, batch_size=256):
        while True:
            x, y = create_batch_function(batch_size)
            yield x, y

    @staticmethod
    def euclidian_distance(vectors):
        (positive, negative) = vectors
        return tf.reduce_mean(tf.square(positive - negative), axis=1)

    def contrastive_loss(self, y_true, y_pred):
        positive, negative = y_pred[:, :self.embedding_size], y_pred[:, self.embedding_size:]
        dist = tf.reduce_mean(tf.square(positive - negative), axis=1)
        return (1 - y_true) * dist + y_true * tf.maximum(0., self.alpha - dist)
        #margin = 1
        #square_pred = K.square(positive - negative)
        #margin_square = K.square(K.maximum(margin - square_pred, 0))
        #return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def train(self, create_batch_function, batch_size, epochs, steps_per_epoch):
        [x_1s, x_2s], ys = self.data_handler.create_validation_pairs()
        validation_data = ([x_1s, x_2s], ys)
        
        history = self.net.fit(
            self.data_generator(create_batch_function, batch_size),
            steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=True,
            validation_data=validation_data)
        
        return history
