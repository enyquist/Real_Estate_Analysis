import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations, metrics
from keras import backend as K


def prep_gpu():
    """
    Function to turn on tensorflow and configure gpu
    :return:
    """
    # Set GPU
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


def create_model(input_size=12, hidden_layers=3):
    """
    Simple wrapper to create a deep learning model using the Keras API
    :param input_size: int, the size of the input (number of features)
    :param hidden_layers: Int, how many hidden layers to create in the model
    :return:
    """
    # Create model
    model = Sequential()

    # Input Layer
    model.add(Dense(units=input_size, input_shape=(input_size,),
                    kernel_initializer='normal', activation=activations.relu, name='Input_layer'))

    # Hidden Layers
    for layer in range(1, hidden_layers + 1):
        model.add(Dense(input_size, kernel_initializer='normal', activation=activations.relu, name=f'Hidden_{layer}'))

    model.add(Dropout(0.05, name=f'Dropout_1'))

    # Output Layer
    model.add(Dense(units=1, kernel_initializer='normal', activation='linear', name='Output_layer'))

    optimizer = Adam(lr=10e-6)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[coeff_determination,
                                                                           metrics.MeanSquaredError()])

    return model


def coeff_determination(y_true, y_pred):
    """
    Custom Keras metric, R2 score
    :param y_true: true label
    :param y_pred: predicted label
    :return:
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())
