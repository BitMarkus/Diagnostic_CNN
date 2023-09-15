import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
def callbacks(checkpoint_path):
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
        initial_value_threshold=.98,)   # minimum/maximum value for saving
    # callbacks.append(model_checkpoint_callback)

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
    if epoch >= 60:
        learning_rate = 1e-06
    if epoch >= 75:
        learning_rate = 1e-07
    if epoch >= 90:
        learning_rate = 1e-08
    # Log learning rate for tensorboard
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate



