from tensorflow.keras.layers import Flatten, Dense, Dropout, Reshape, Conv2D, MaxPooling2D, Lambda 
from tensorflow.keras.models import Sequential
from tensorflow.math import l2_normalize


class ModelHandler:
    # Constructor.
    def __init__(self, model_number, embedding_size, input_feature_dim):
        self.embedding_size = embedding_size
        self.input_feature_dim = input_feature_dim
        self.input_feature_size = input_feature_dim[0]*input_feature_dim[1]
        self.model_number = model_number
        self.model = self.create_model()

    # Select one of the base embedding models.
    def create_model(self):
        if self.model_number == 0:
            model = Sequential()
            model.add(Dense(64, activation='relu', input_shape=(self.input_feature_size,)))
            model.add(Dense(self.embedding_size, activation='sigmoid'))
        if self.model_number == 1:
            model = Sequential()
            model.add(Dense(256, activation='relu', input_shape=(self.input_feature_size,)))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(self.embedding_size, activation='sigmoid'))
        if self.model_number == 2:
            model = Sequential()
            model.add(Reshape(self.input_feature_dim, input_shape=(self.input_feature_size,)))
        if self.model_number == 3:
            model = Sequential()
            model.add(Reshape(self.input_feature_dim, input_shape=(self.input_feature_size,)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(self.embedding_size, activation='sigmoid'))
        if self.model_number == 4:
            model = Sequential()
            model.add(Reshape(self.input_feature_dim, input_shape=(self.input_feature_size,)))
            model.add(Conv2D(64, (10,10), activation='relu', input_shape=self.input_feature_dim))
            model.add(MaxPooling2D())
            model.add(Conv2D(128, (7,7), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(self.embedding_size, activation='sigmoid'))
        if self.model_number == 5:
            model = Sequential()
            model.add(Reshape(self.input_feature_dim, input_shape=(self.input_feature_size,)))
            model.add(Conv2D(64, (10,10), activation='relu', input_shape=self.input_feature_dim))
            model.add(MaxPooling2D())
            model.add(Conv2D(128, (7,7), activation='relu'))
            model.add(MaxPooling2D())
            model.add(Flatten())
            model.add(Dense(self.embedding_size, activation=None)) # no activation on the final dense layer
            Lambda(lambda x: l2_normalize(x, axis=1)) # L2 normalize embeddings
            # Not sure if this normalization layer is necessary...also activation in the dense layer used to be sigmoid
            
        return model

