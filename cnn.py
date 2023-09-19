from tensorflow import keras
from keras import layers, regularizers

# Define function for model creation
def cnn_model(ds_shape, dropout, l2_weight_decay=0, num_classes=2):
    # Input shape
    inputs = keras.Input(shape=ds_shape)
    ###############
    # CNN BLOCK 1 #
    ###############
    # CNN LAYER 1.1:
    x = layers.Conv2D(
        62, 
        3, 
        padding='same',
        name='cnn_1_1'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # CNN LAYER 1.2:
    x = layers.Conv2D(
        62, 
        3, 
        padding='same',
        name='cnn_1_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
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
    x = keras.activations.relu(x)
    # CNN LAYER 2.2:
    x = layers.Conv2D(
        128, 
        3, 
        padding='same',
        name='cnn_2_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
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
    x = keras.activations.relu(x)
    # CNN LAYER 3.2:
    x = layers.Conv2D(
        256, 
        3, 
        padding='same',
        name='cnn_3_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # CNN LAYER 3.3:
    x = layers.Conv2D(
        256, 
        3, 
        padding='same',
        name='cnn_3_3'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
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
    x = keras.activations.relu(x)
    # CNN LAYER 4.2:
    x = layers.Conv2D(
        512, 
        3, 
        padding='same',
        name='cnn_4_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # CNN LAYER 4.3:
    x = layers.Conv2D(
        512, 
        3, 
        padding='same',
        name='cnn_4_3'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D(2)(x)
    ###############
    # CNN BLOCK 5 #
    ###############
    # CNN LAYER 5.1:
    x = layers.Conv2D(
        512, 
        3, 
        padding='same',
        name='cnn_5_1'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # CNN LAYER 5.2:
    x = layers.Conv2D(
        512, 
        3, 
        padding='same',
        name='cnn_5_2'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    # CNN LAYER 5.3:
    x = layers.Conv2D(
        512, 
        3, 
        padding='same',
        name='cnn_5_3'
    )(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)

    #############
    # FC LAYERS #
    #############
    # FC LAYER 1:
    if(l2_weight_decay == 0):
        x = layers.Dense(
            1024, 
            activation='relu', 
            name='fc_1'                       
        )(x)
    else:
        x = layers.Dense(
            1024, 
            activation='relu',                        
            kernel_regularizer=regularizers.l2(l2_weight_decay),
            name='fc_1'   
        )(x)
    x = layers.Dropout(dropout)(x)
    # FC LAYER 2:
    if(l2_weight_decay == 0):
        x = layers.Dense(
            512, 
            activation='relu', 
            name='fc_2'                       
        )(x)
    else:
        x = layers.Dense(
            512, 
            activation='relu',                        
            kernel_regularizer=regularizers.l2(l2_weight_decay),
            name='fc_2'   
        )(x)
    x = layers.Dropout(dropout)(x)

    ##############
    # CLASSIFYER #
    ##############
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name='output',
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model