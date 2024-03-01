import matplotlib.pyplot as plt
from tensorflow import keras
import math
import os
from datetime import datetime
from fcn import load_img, get_cnn_layer_info

# Function to retain program execution while a plot is shown
# Can be used instead of plt.show()
# https://stackoverflow.com/questions/65951965/when-i-plot-something-in-python-the-programs-execution-stops-until-i-close-the-p
# It is important that the last line of the main code is plt.show()
# or the plot will close when the program execution is finished
def show_plot_exec():
    plt.show(block=False)
    plt.pause(0.001)  

# Function to save the acc/loss plot
def save_metrics_plot(eval_history, plot_path, seed):
    # Create folder if not exists
    if not plot_path.exists():
        os.mkdir(plot_path)
    # Evaluation accuracy (for filename)
    eval_acc = eval_history[1]
    # Get date and time
    date_time = datetime.now().strftime("%Y_%m_%d-%H_%M")
    # Generate filename
    filename = f"{date_time}-tacc{eval_acc:.2f}-seed{seed}.png"
    # Save plot
    plt.savefig(str(plot_path) + '/' + filename, bbox_inches='tight')

# Prints accuracy and loss after training
def plot_metrics(train_history, eval_history, plot_path, seed, show_plot=True, save_plot=False):
    # Accuracy
    acc = train_history.history['accuracy']
    val_acc = train_history.history['val_accuracy']
    # Loss
    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    # Learning rate
    lr = train_history.history['learning_rate']
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
        show_plot_exec()

# Shows a single image using matplotlib in false colors
def plot_image(data_path, vis_path, img_class, img_name, save_plot=True, show_plot=False):
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
        show_plot_exec()

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
    show_plot_exec()

# Function to plot a feature map of a single layer 
# https://insights.willogy.io/tensorflow-insights-part-3-visualizations/
def plot_feature_maps_of_single_layer(feature_maps, 
                                      vis_path, 
                                      img_name, 
                                      img_class,
                                      layer_info, 
                                      num_rows, 
                                      num_cols, 
                                      save_plot=False, 
                                      show_plot=True):
    # Set title
    plt.figure(figsize=(18, 13))
    title = f'Image: {img_name}, Class: {img_class}\n'
    title += f'Feature maps of layer {layer_info["index"]} ({layer_info["name"]}): '
    title += f'{feature_maps.shape[1]}x{feature_maps.shape[2]} px'
    plt.suptitle(title, fontsize=15)
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
        show_plot_exec()

# Function to plot feature maps of all cnn layers
# https://insights.willogy.io/tensorflow-insights-part-3-visualizations/
def plot_feature_maps_of_multiple_layers(model,
                                         data_path,
                                         vis_path, 
                                         img_class,
                                         img_name,
                                         num_rows=3, 
                                         num_cols=3, 
                                         save_plot=False, 
                                         show_plot=True):
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
def plot_img_batch(batch_size, ds, vis_path, show_plot=True, save_plot=False):
    # Get class names
    class_names = ds.class_names
    # Calculate number of rows/columns
    nr_row_col = math.ceil(math.sqrt(batch_size))
    # print(f"cols/rows: {nr_row_col}\n")
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(batch_size):
            plt.subplot(nr_row_col, nr_row_col, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap='inferno')
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.tight_layout()
    # Save plot
    filename = f'batch0_dataset'
    if(save_plot):
        plt.savefig(str(vis_path) + '/' + filename, bbox_inches='tight')
    # Show plot
    if(show_plot):
        show_plot_exec()
