# https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception
# https://maelfabien.github.io/deeplearning/xception/#the-limits-of-convolutions
# https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
# https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568

# from keras import layers
# from keras.models import Model
from keras.applications import Xception

# Define function for model creation
def xception_model(ds_shape, num_classes=2):
    model = Xception(
        include_top=True,
        weights=None,
        input_shape=ds_shape,
        pooling=None,
        classes=num_classes,
        classifier_activation='softmax'
    )
    return model
