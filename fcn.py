import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

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

# Prepare training, validation and test dataset
def get_ds(data_dir, batch_size, img_height, img_width, val_split, seed, categories):
    # Training dataset:
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,                               # directory with training images, classes in seperate folders
        labels='inferred',                      # lables are taken from subfolder names
        label_mode="int",                       # OR categorical, binary
        class_names=categories,
        color_mode='grayscale',                 # OR rgb
        batch_size=batch_size,
        image_size=(img_height, img_width),     # images will be reshaped if not in this size
        shuffle=True,
        seed=seed,                               # set seed if training should be the same any time it runs
        validation_split=val_split,             # images used for validation
        subset='training',
    )   
    # Validation dataset:
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,                              
        labels='inferred',                     
        label_mode="int",                       
        class_names=categories,
        color_mode='grayscale',                     
        batch_size=batch_size,
        image_size=(img_height, img_width),    
        shuffle=True,
        seed=seed,                               
        validation_split=val_split,             
        subset='validation',
    )
    # Test dataset:
    # source: https://stackoverflow.com/questions/66036271/splitting-a-tensorflow-dataset-into-training-test-and-validation-sets-from-ker
    # determine how many batches of data are available in the validation set:
    num_batches = tf.data.experimental.cardinality(ds_validation)
    # move the two-third of them (2/3 of 30% = 20%) to a test set
    # // = rounded to the next smallest whole number = integer division
    ds_test = ds_validation.take((2*num_batches) // 3)
    ds_validation = ds_validation.skip((2*num_batches) // 3)
    return ds_train, ds_validation, ds_test

# Normalize images from 8 bit to values between 0-1
def normalize_img(image, label):
    img = tf.cast(image, tf.float32)/255.0
    return img, label

# Convert RGB image to grayscale
def rgb_to_gray_img(image, label):
    img = tf.image.rgb_to_grayscale(image)
    return img, label

# Function for data augmentation
def augment_img(image, label):
    # Further data augmentation
    # image = tf.image.random_brightness(image, max_delta=0.1)
    # image = tf.image.random_contrast(image, lower=0.2, upper=0.5)
    image = tf.image.random_flip_left_right(image)  # is done in 50% of the cases
    image = tf.image.random_flip_up_down(image)
    # image = tf.image.per_image_standardization(image)
    return image, label

# Function for data tuning
def tune_img(ds_train, ds_validation, ds_test, autotune):
    # Prepare dataset and configure dataset for performance
    # Setup for training dataset:
    ds_train = ds_train.map(normalize_img, num_parallel_calls=autotune)
    ds_train = ds_train.cache()
    ds_train = ds_train.prefetch(buffer_size=autotune)
    # Setup for validation dataset:
    ds_validation = ds_validation.map(normalize_img, num_parallel_calls=autotune)
    ds_validation = ds_validation.cache() # Is this necessary here?
    ds_validation = ds_validation.prefetch(buffer_size=autotune)
    # Setup for test dataset:
    ds_test = ds_test.map(normalize_img, num_parallel_calls=autotune)
    ds_test = ds_test.cache() # Is this necessary here?
    ds_test = ds_test.prefetch(buffer_size=autotune)
    return ds_train, ds_validation, ds_test

# Function for callbacks
def get_callbacks(checkpoint_path):
    callbacks = []
    
    # Save best weights as checkpoint (highest validation accuracy):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,       # Path to checkpoint file (not only folder)
        save_weights_only=True,         # False = whole model will be saved
        monitor='val_accuracy',         # Value to monitor
        mode='max',                     # max, min, auto fpr value to monitor
        save_best_only=True,            # save only the best model/weights
        verbose=1,                      # show messages
        save_freq='epoch',              # check after every epoch
        initial_value_threshold=.96,)   # minimum/maximum value for saving
    callbacks.append(model_checkpoint_callback)

    # Early stopping:
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=20,            # 10-15
        start_from_epoch=40     # 40
    )
    # callbacks.append(early_stopping_callback)

    # Tensorboard:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        # 1) Activate anaconda environment for tensorflow (tf_gpu)
        # 2) Go to folder where tensorflow script is (cd /home/markus/tensorflow/projects/Diagnostic_CNN)
        # 3) Type into console: tensorboard --logdir logs (<- Folder specified below)
        # 4) Go to shown URL: http://localhost:6006
        # 5) Go to "scalars" to see accuracy and loss
        log_dir="logs",             # Directory to store log files
        histogram_freq=0,           # frequency (in epochs) at which to compute activation histograms for the layers of the model
        write_graph=True,           # whether to visualize the graph in Tensorboard
        write_grads=False,          # whether to visualize gradient histograms in TensorBoard (histogram_freq must be greater than 0)
        write_images=False,         # whether to write model weights to visualize as image in Tensorboard
        update_freq="epoch",        # 'batch' or 'epoch' or integer.
    )
    callbacks.append(tensorboard_callback)

    # Learning rate scheduler: reduces learning rate dependent on epoch index -> own function!
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(
        lr_scheduler, 
        verbose=1
    )
    callbacks.append(lr_scheduler_callback)

    return callbacks

# Function for reducing the learning rate dependent on the epoch
# For callback 'lr_scheduler_callback'
def lr_scheduler(epoch):
    learning_rate = 1e-05
    if epoch >= 30:
        learning_rate = 1e-06
    if epoch >= 50:
        learning_rate = 1e-07
    if epoch >= 70:
        learning_rate = 1e-08
    # Log learning rate for tensorboard
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

# Function loads a single image for prdictions
# It also normalizes the image
def load_img(pth, category, name):
    # Load image
    img = keras.utils.load_img(
        pth / category / name,
        color_mode='grayscale', 
        target_size=None,
    )
    # Convert PIL object to numpy array
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # Normalize pixel values to 0-1
    img = tf.cast(img, tf.float32)/255.0
    return img

# Function predicts a single image and prints output
def predict_single_img(model, data_path, subfolder, img_name, class_names):
    # load image
    img = load_img(data_path, subfolder, img_name)
    # Predict probabilities: return of a 2D numpy array (why 2D?)
    print(f"Predict class of image \"{img_name}\":")
    probabilities = model.predict(img, verbose=1)
    # Connect probability with the respective class label
    class_index = np.argmax(probabilities, axis=-1)
    pred_class_name = class_names[class_index[0]]
    pred_probability = probabilities[0][class_index[0]]*100
    print(f"Image {img_name} belongs to class \"{pred_class_name}\" ({pred_probability:.2f}%)")   

# Function counts the number of cnn layers in a model
def num_cnn_layers(model):
    count = 0
    for _, layer in enumerate(model.layers):
        if 'cnn' in layer.name: 
            count += 1
    return count

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

# Function to retain program execution while a plot is shown
# Can be used instead of plt.show()
# https://stackoverflow.com/questions/65951965/when-i-plot-something-in-python-the-programs-execution-stops-until-i-close-the-p
# It is important that the last line of the main code is plt.show()
# or the plot will close when the program execution is finished
def show_plot_exec():
    plt.show(block=False)
    plt.pause(0.001)  

# Function returns a list with all class names
# = all subfolders in the data folder
# https://www.techiedelight.com/list-all-subdirectories-in-directory-python/
def get_class_names(data_dir):
    class_list = []
    for file in os.listdir(data_dir):
        d = os.path.join(data_dir, file)
        # Only folders, no files
        if os.path.isdir(d): 
            class_list.append(file)
    return class_list



