from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras import utils

def add_default_block(x, kernel_filters):
    s = layers.Conv2D(kernel_filters, (1,1), padding='same')(x)
    s = layers.BatchNormalization()(s)
    s = layers.Activation('relu')(s)

    x = layers.Conv2D(kernel_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(kernel_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Add()([x, s])
    x = layers.MaxPooling2D()(x)

    return x

def ResNet18(input_shape=None, include_top=True, input_tensor=None, n_classes=10):

    if input_shape is None:
        input_shape = (128, 128, 1)

    if input_tensor is None:
        input_tensor = keras.Input(input_shape)

    x = input_tensor

    # first (non-default) block
    x = layers.Conv2D(32, (7, 7), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3,3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 2nd-5th (default) blocks
    x = add_default_block(x, 64)
    x = add_default_block(x, 128)
    x = add_default_block(x, 256)
    x = add_default_block(x, 512)

    # Classification
    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)  
        x = layers.Dense(n_classes, activation='sigmoid' if n_classes == 1 else 'softmax')(x)   

    outputs = x
    inputs = utils.layer_utils.get_source_inputs(input_tensor)[0]

    model = keras.Model(inputs, outputs)
    return model

if __name__ == '__main__':
    model = ResNet18((256, 256, 1))
    model.summary()
    print(model.output)
    print(model.output.shape)