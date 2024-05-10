# https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16

from keras.applications import VGG16
from keras.models import Model
from keras import layers

# Define function for model creation
def vgg16_model(ds_shape, num_classes=2):
    vgg16 = VGG16(
        include_top=False,
        weights=None,
        input_shape=ds_shape,
        pooling=None,
        classes=None,
        classifier_activation=None
    )
    x = vgg16.output
    x = layers.GlobalAveragePooling2D()(x)   
    # Add classifier
    if(num_classes == 2):
        out = layers.Dense(1, activation='sigmoid', name='output',)(x)
    elif(num_classes > 2):
        out = layers.Dense(num_classes, activation='softmax', name='output',)(x)
    model = Model(inputs = vgg16.input, outputs = out)
    return model
