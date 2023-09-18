import matplotlib.pyplot as plt
from tensorflow import keras

# Function counts the number of cnn layers in a model
def num_cnn_layers(model):
    count = 0
    for id, layer in enumerate(model.layers):
        if 'cnn' in layer.name: 
            count += 1
    return count

# https://insights.willogy.io/tensorflow-insights-part-3-visualizations/
# function plot_filters_of_a_layer for visualizing filters of a layer
# works only with grayscale pictures!
def plot_filters_of_layers(model, num_filters):
    # Get weights (and bias) as list
    weights_layers = []
    for _, layer in enumerate(model.layers):
        # If a layer is not a convolutional layer
        if 'cnn' in layer.name: 
            # Get weights
            filters_weights, biases_weights = layer.get_weights()
            # Normalizing filters to 0-1
            filters_max, filters_min = filters_weights.max(), filters_weights.min()
            filters_weights = (filters_weights - filters_min)/(filters_max - filters_min)
            # Append to filter list
            weights_layers.append(filters_weights)
    # Get number of cnn layers
    num_cnn_layers = len(weights_layers)
    # print(weights_layers)
    # print(num_cnn_layers)
    # plot first few filters
    plt.figure(figsize=(12, 12))
    ix = 1
    for i in range(num_cnn_layers):
        for j in range(num_filters):
            # get the filter
            f = weights_layers[i][:, :, :, j]
            # specify subplot and turn of axis
            ax = plt.subplot(num_cnn_layers, num_filters, ix)
            ax.set_xticks([])
            ax.set_yticks([])    
            # plot filter channel in grayscale
            plt.imshow(f[:, :, i], cmap='gray')
            ix += 1    
    # show the figure
    plt.tight_layout()
    plt.show()

# Function to plot feature maps
def plot_feature_maps_of_a_layer(feature_maps, num_rows, num_cols):
    plt.figure(figsize=(12, 12))
    ix = 1
    for _ in range(num_rows): # Plot 8 images in 2 row-4 column table
        for _ in range(num_cols):
            ax = plt.subplot(num_rows, num_cols, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1 # increase the index of the last dimension to visualize 8 feature maps
    plt.tight_layout()
    plt.show()

def plot_feature_maps_of_layers(model, img, num_rows, num_cols):
    # Get indices of all convolution layers
    conv_layers_index = []
    for idx, layer in enumerate(model.layers):
        if 'cnn' in layer.name:
            conv_layers_index.append(idx)
    # print(conv_layers_index)

    list_of_outputs = [model.layers[idx].output for idx in conv_layers_index]
    model_tmp = keras.Model(inputs=model.inputs, outputs=list_of_outputs)
    # model_tmp.summary()
    feature_maps = model_tmp.predict(img) 
    for feature_map in feature_maps:
        print('[*] feature_map.shape: ', feature_map.shape)
        plot_feature_maps_of_a_layer(feature_map, num_rows, num_cols)
