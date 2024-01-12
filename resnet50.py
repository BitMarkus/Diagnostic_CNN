# https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/cv_resnet50.html
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-resnet-from-scratch-with-tensorflow-2-and-keras.md
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
# https://www.tensorflow.org/tutorials/images/transfer_learning
# https://github.com/ovh/ai-training-examples/blob/main/notebooks/computer-vision/image-classification/tensorflow/resnet50/notebook-resnet-transfer-learning-image-classification.ipynb
# https://stackabuse.com/dont-use-flatten-global-pooling-for-cnns-with-tensorflow-and-keras/
# https://www.quora.com/Why-was-global-average-pooling-used-instead-of-a-fully-connected-layer-in-GoogLeNet-and-how-was-it-different

from keras import layers
from keras.models import Model
from keras.applications import ResNet50

# Define function for model creation
def resnet_model(ds_shape, dropout, num_classes=2):
    # ResNet50 base model without glogal average pooling and classifier
    resnet_50 = ResNet50(
        include_top=False,
        weights=None,
        input_shape=ds_shape,
        pooling=None,
    )
    # print(resnet_50.summary())
    x = resnet_50.output
    """
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', name='dense_0')(x) 
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(512, activation='relu', name='dense_1')(x) 
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu', name='dense_2')(x) 
    x = layers.Dropout(dropout)(x)
    """
    x = layers.GlobalAveragePooling2D()(x)   
    # Classifier:
    predictions = layers.Dense(num_classes, activation='softmax', name='output',)(x)
    model = Model(inputs = resnet_50.input, outputs = predictions)

    return model
