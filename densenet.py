# https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet121

from keras.applications import DenseNet121, DenseNet201

# Define function for model creation
def densenet_model(ds_shape, num_classes=2):
    # ResNet50 base model without glogal average pooling and classifier
    # model = DenseNet201(
    model = DenseNet121(
        include_top=True,
        weights=None,
        input_shape=ds_shape,
        pooling=None,
        classes=num_classes,
        classifier_activation='softmax'
    )
    # print(model.summary())

    return model