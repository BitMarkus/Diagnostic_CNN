import matplotlib.pyplot as plt
from tensorflow import keras
import math
# from matplotlib import pylab
from fcn import save_metrics_plot, load_img

# Function counts the number of cnn layers in a model
def num_cnn_layers(model):
    count = 0
    for _, layer in enumerate(model.layers):
        if 'cnn' in layer.name: 
            count += 1
    return count

# https://insights.willogy.io/tensorflow-insights-part-3-visualizations/
# Function plot_filters_of_a_layer for visualizing filters of a layer
# works only with grayscale pictures!
def plot_filters_of_layers(model, num_filters):
    # Get weights (and bias) as list
    weights_layers = []
    for _, layer in enumerate(model.layers):
        # If a layer is not a convolutional layer
        if 'cnn' in layer.name: 
            # Get weights
            filters_weights, _ = layer.get_weights()
            # Normalizing weights to 0-1
            filters_max, filters_min = filters_weights.max(), filters_weights.min()
            filters_weights = (filters_weights - filters_min)/(filters_max - filters_min)
            # Append to filter list
            weights_layers.append(filters_weights)
    # Get number of cnn layers
    num_cnn_layers = len(weights_layers)

    # Plot first few filters
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

# Function returns a dict with all indices and layer names
#  of all cnn layers in the model as list
def get_cnn_layer_info(model):
    conv_layers_index = []
    for idx, layer in enumerate(model.layers):
        if 'cnn' in layer.name:
            conv_layers_index.append(
                {
                    "index": idx,
                    "name": layer.name,
                }
            )
    return conv_layers_index

# https://insights.willogy.io/tensorflow-insights-part-3-visualizations/
# Function to plot a feature map of a single layer 
def plot_feature_maps_of_single_layer(feature_maps, 
                                      vis_path, 
                                      img_name, 
                                      img_class,
                                      layer_info, 
                                      num_rows, 
                                      num_cols, 
                                      save_plot=False, 
                                      show_plot=False):
    # Set title
    plt.figure(figsize=(18, 13))
    title = f'Image: {img_name}, Class: {img_class}\n'
    title += f'Feature maps of layer {layer_info["index"]} ({layer_info["name"]}): '
    title += f'{feature_maps.shape[1]}x{feature_maps.shape[2]} px'
    plt.suptitle(title, fontsize=15)
    # Set windows title
    # fig = pylab.gcf()
    # fig.canvas.manager.set_window_title(title)
    index = 1
    # Only num_rows x num_cols feature maps will be displayed
    for _ in range(num_rows):
        for _ in range(num_cols):
            plt.subplot(num_rows, num_cols, index)
            plt.title(f'map {index} of {feature_maps.shape[3]}')
            # cmap='gray' for normal b/w images
            plt.imshow(feature_maps[0, :, :, index-1], cmap='inferno')
            # Colorbar height: https://www.geeksforgeeks.org/set-matplotlib-colorbar-size-to-match-graph/
            # Calculate (height_of_image / width_of_image)
            img_ratio = feature_maps.shape[1]/feature_maps.shape[2]
            plt.colorbar(fraction=0.047*img_ratio)
            index += 1
    plt.tight_layout()
    # Save plot
    filename = f'{layer_info["name"]}_{img_class}_{img_name}'
    if(save_plot):
        plt.savefig(str(vis_path) + '/' + filename, bbox_inches='tight')
    # Show plot
    if(show_plot):
        plt.show()

# https://insights.willogy.io/tensorflow-insights-part-3-visualizations/
# Function to plot feature maps of all cnn layers
# Only the ones from the end of each cnn block will be shown
def plot_feature_maps_of_multiple_layers(model,
                                         data_path,
                                         vis_path, 
                                         img_class,
                                         img_name,
                                         num_rows=3, 
                                         num_cols=3, 
                                         save_plot=False, 
                                         show_plot=False):
    # load image
    img = load_img(data_path, img_class, img_name)
    # Get indices of ALL convolution layers
    conv_layers = get_cnn_layer_info(model)
    # Get list with desired output layers
    list_of_outputs = []
    for cnnlayer in conv_layers:
        list_of_outputs.append(model.layers[cnnlayer["index"]].output)
    # Generate temporary model using outputs of cnn lyers
    model_tmp = keras.Model(inputs=model.inputs, outputs=list_of_outputs)
    # Send image through network
    feature_maps = model_tmp.predict(img, verbose=0) 
    # Plot feature maps
    idx = 0
    print("\nPrint feature maps:")
    for feature_map in feature_maps:
        print(f'>> Layer {conv_layers[idx]["index"]} ({conv_layers[idx]["name"]}) shape:', feature_map.shape)
        # Plot all maps
        plot_feature_maps_of_single_layer(feature_map, 
                                          vis_path, 
                                          img_name, 
                                          img_class,
                                          conv_layers[idx], 
                                          num_rows, 
                                          num_cols,
                                          save_plot=save_plot, 
                                          show_plot=show_plot)
        idx += 1

# Prints first batch of images from the training dataset
def print_img_batch(batch_size, ds, class_names):
    # Get class names
    # class_names = ds.class_names
    # Calculate number of rows/columns
    nr_row_col = math.ceil(math.sqrt(batch_size))
    # print(f"cols/rows: {nr_row_col}\n")
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(batch_size):
            plt.subplot(nr_row_col, nr_row_col, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

# Prints accuracy and loss after training
def create_metrics_plot(train_history, eval_history, plot_path, seed, show_plot=True, save_plot=True):
    # Accuracy
    acc = train_history.history['accuracy']
    val_acc = train_history.history['val_accuracy']
    # Loss
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    # Learning rate
    lr = train_history.history['lr']
    # Number of epochs
    epochs_range = range(1, len(acc) + 1)
    # Draw plots
    plt.figure(figsize=(15, 5))
    # Accuracy plot:
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='green')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='red')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    # Loss plot:
    plt.subplot(1, 3, 2)
    # Set the range of y-axis
    plt.ylim(0, 5)
    plt.plot(epochs_range, loss, label='Training Loss', color='green')
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='red')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # Learning rate plot:
    plt.subplot(1, 3, 3)
    # convert y-axis to Logarithmic scale
    plt.yscale("log")
    plt.plot(epochs_range, lr, label='Learning Rate', color='blue')
    plt.legend(loc='upper right')
    plt.title('Learning Rate')
    # Reduce unnecessary whitespaces around the plots
    # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    plt.tight_layout()
    # Save plot
    if(save_plot):
        save_metrics_plot(eval_history, plot_path, seed)
    # Show plot
    if(show_plot):
        plt.show()

def plot_image(data_path, vis_path, img_class, img_name, save_plot=True, show_plot=True):
    # load image
    img = load_img(data_path, img_class, img_name)
    # prepare plot
    plt.figure(figsize=(10, 10))
    title = f'Image: {img_name}, Class: {img_class}, Size: '
    title += f'{img.shape[1]}x{img.shape[2]} px'
    plt.title(title, fontsize=15)
    # Reduce dimensions of image tensor
    plt.imshow(img[-1, :, :, :], cmap='inferno')
    # Colorbar height: https://www.geeksforgeeks.org/set-matplotlib-colorbar-size-to-match-graph/
    # Calculate (height_of_image / width_of_image)
    img_ratio = img.shape[1]/img.shape[2]
    plt.colorbar(fraction=0.047*img_ratio)
    plt.tight_layout()
    # Save plot
    filename = f'cnn_0_input_{img_class}_{img_name}'
    if(save_plot):
        plt.savefig(str(vis_path) + '/' + filename, bbox_inches='tight')
    # Show plot
    if(show_plot):
        plt.show()