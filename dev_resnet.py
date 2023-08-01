from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras import utils

def conv2plus1d(x, filters, kernel_size, maxpool=True, padding='same'):
    t = layers.Conv3D(filters, (kernel_size[0], 1, 1), padding=padding)(x)
    t = layers.BatchNormalization()(t)
    t = layers.Activation('relu')(t)

    x = layers.Conv3D(filters, (1,kernel_size[1],kernel_size[2]), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(filters, (1,kernel_size[1],kernel_size[2]), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Add()([x, t])
    if maxpool:
        x = layers.MaxPooling3D(pool_size=(1,2,2))(x)

    return x

def stem2dplus1d(x):
    x = layers.Conv3D(32, (1,7,7), strides=(1,2,2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    t = layers.Conv3D(32, (3,1,1), padding='same')(x)
    t = layers.BatchNormalization()(t)
    t = layers.Activation('relu')(t)

    x = layers.Conv3D(32, (1,3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Add()([x,t])
    x = layers.MaxPooling3D(pool_size=(1,2,2))(x)
    return x

def DevResNet(input_shape=None, include_top=True, input_tensor=None, n_classes=10):
    '''
    Create an instance of Dev-ResNet

    Initiate a keras model instance of Dev-ResNet - a 3D CNN for detecting developmental events
    in developing animals.

    Parameters
    ----------
    input_shape : tuple
        Desired input shape of model (length, width, height, channels). Note if you use the weights 
        for Lymnaea stagnalis you will need an input shape of (12,128,128,1).
    include_top : bool
        Whether to include a final classification layer.
    input_tensor : tf.Tensor
        An input tensor to override the creation of the input tensor within the function.
    n_classes : int
        Number of classes for classification if include_top=True.
    
    Returns
    -------
    model : keras.Model
        Keras model instance
        
    '''

    if input_shape is None:
        input_shape = (12, 128, 128, 1)

    if input_tensor is None:
        input_tensor = keras.Input(input_shape)

    x = input_tensor
    x = layers.TimeDistributed(layers.Rescaling(1.0 / 255))(x)

    # Stem
    x = stem2dplus1d(x)

    # Residual blocks
    x = conv2plus1d(x, 64, (3,3,3), maxpool=True)
    x = conv2plus1d(x, 128, (3,3,3), maxpool=True)
    x = conv2plus1d(x, 256, (3,3,3), maxpool=True)
    x = conv2plus1d(x, 512, (3,3,3), maxpool=True)

    # Classification
    if include_top:
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.BatchNormalization()(x)  
        x = layers.Dense(n_classes, activation='sigmoid' if n_classes == 1 else 'softmax')(x)   

    outputs = x
    inputs = utils.layer_utils.get_source_inputs(input_tensor)[0]

    return keras.Model(inputs=inputs, outputs=outputs)  


if __name__ == '__main__':
    model = DevResNet((12, 128, 128, 1))
    model.summary()