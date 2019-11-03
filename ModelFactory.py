from keras.models import Sequential
from keras.layers import Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.initializers import TruncatedNormal
from ConfigurationFactory import ConfigurationFactory

def create_fpool3_model():
    configuration = ConfigurationFactory.CreateConfiguration()

    model = Sequential()

    p_pooling_in_time = 1
    q_pooling_in_frequency = 3

    model.add(Convolution2D(input_shape=[40,32,1], filters=64, kernel_size=[20, 8], strides=[1, 1],
        padding='same', kernel_initializer=TruncatedNormal(stddev=0.05), activation='relu'))

    model.add(Dropout(rate=0.5))

    model.add(MaxPooling2D(pool_size=[p_pooling_in_time, q_pooling_in_frequency],
        strides=[p_pooling_in_time, q_pooling_in_frequency], padding='same'))

    if configuration.use_normalization:
        model.add(BatchNormalization())

    model.add(Convolution2D(filters=64, kernel_size=[10, 4], strides=[1, 1], padding='same',
        kernel_initializer=TruncatedNormal(stddev=0.01), activation='relu'))

    model.add(Dropout(rate=0.5))

    # May need batch flatten
    model.add(Flatten())

    model.add(Dense(32, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.01)))
    model.add(Dense(128, activation='relu', kernel_initializer=TruncatedNormal(stddev=0.01)))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=0.01)))

    print(model.summary())
    return model

class ModelFactory:
    pass
