# https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception
# https://maelfabien.github.io/deeplearning/xception/#the-limits-of-convolutions
# https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
# https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568

from keras.applications import Xception
from keras.models import Model
from keras import layers

# Define function for model creation
def xception_model(ds_shape, num_classes=2):
    xcept = Xception(
        include_top=False,
        weights=None,
        input_shape=ds_shape,
        pooling=None,
        classes=None,
        classifier_activation=None
    )
    x = xcept.output
    x = layers.GlobalAveragePooling2D()(x)   
    # Add classifier
    if(num_classes == 2):
        out = layers.Dense(1, activation='sigmoid', name='output',)(x)
    elif(num_classes > 2):
        out = layers.Dense(num_classes, activation='softmax', name='output',)(x)
    model = Model(inputs = xcept.input, outputs = out)
    return model
