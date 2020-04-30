# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model


def basic_cnn_model_with_regularization(input_shape, layer_dimensions, should_add_dropout=False):
    """
    Add batch normalization and appropriate weight initialization.
    """
    kernel_initializer = 'he_uniform'
    input_layer = Input(shape=input_shape)
    current_layer = input_layer
    for layer_dimension in layer_dimensions:
        layer = Dense(layer_dimension, activation='relu',
                      kernel_initializer=kernel_initializer, use_bias=False)(current_layer)
        current_layer = BatchNormalization()(layer)

    if should_add_dropout:
        current_layer = Dropout(.2)(current_layer)

    output = Dense(1, kernel_initializer='normal', activation='linear')(current_layer)

    return Model(inputs=input_layer, outputs=output)
