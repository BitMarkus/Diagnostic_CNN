# https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet121

from keras.applications import DenseNet121, DenseNet201

# Define function for model creation
def densenet_model(ds_shape, num_classes=2):
    model = DenseNet201(
        include_top=True,
        weights=None,
        input_shape=ds_shape,
        classes=num_classes,
        classifier_activation='softmax'
    )
    # print(model.summary())

    return model