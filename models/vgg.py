from tensorflow import keras
from keras import layers, regularizers
from keras.activations import relu

# Define function for model creation
def vgg_model(ds_shape, dropout, l2_weight_decay=0, num_classes=2):
    # Input shape
    inputs = keras.Input(shape=ds_shape)
    ###############
    # CNN BLOCK 1 #
    ###############
    # CNN LAYER 1.1:
    x = layers.Conv2D(
        64, 
        3, 
        padding='same',
        name='cnn_1_1'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    # CNN LAYER 1.2:
    x = layers.Conv2D(
        64, 
        3, 
        padding='same',
        name='cnn_1_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    x = layers.MaxPooling2D(2)(x)
    ###############
    # CNN BLOCK 2 #
    ###############
    # CNN LAYER 2.1:
    x = layers.Conv2D(
        128, 
        3, 
        padding='same',
        name='cnn_2_1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    # CNN LAYER 2.2:
    x = layers.Conv2D(
        128, 
        3, 
        padding='same',
        name='cnn_2_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    x = layers.MaxPooling2D(2)(x)
    ###############
    # CNN BLOCK 3 #
    ###############
    # CNN LAYER 3.1:
    x = layers.Conv2D(
        256, 
        3, 
        padding='same',
        name='cnn_3_1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    # CNN LAYER 3.2:
    x = layers.Conv2D(
        256, 
        3, 
        padding='same',
        name='cnn_3_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    # CNN LAYER 3.3:
    x = layers.Conv2D(
        256, 
        3, 
        padding='same',
        name='cnn_3_3'
    )(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    x = layers.MaxPooling2D(2)(x)
    ###############
    # CNN BLOCK 4 #
    ###############
    # CNN LAYER 4.1:
    x = layers.Conv2D(
        512, 
        3, 
        padding='same',
        name='cnn_4_1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    # CNN LAYER 4.2:
    x = layers.Conv2D(
        512, 
        3, 
        padding='same',
        name='cnn_4_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = relu(x)
    # x = layers.MaxPooling2D(2)(x)

    # Global average pooling (instead of flatten)
    x = layers.GlobalAveragePooling2D()(x)  

    ##############
    # CLASSIFYER #
    ##############

    if(num_classes == 2):
        outputs = layers.Dense(1, activation='sigmoid', name='output',)(x)
    elif(num_classes > 2):
        outputs = layers.Dense(num_classes, activation='softmax', name='output',)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model