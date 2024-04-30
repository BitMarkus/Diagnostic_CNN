# Tensorboard:
# 1) Activate anaconda environment for tensorflow (tf_gpu)
# 2) Go to folder where tensorflow script is (cd /home/markus/tensorflow/projects/Diagnostic_CNN)
# 3) Type into console: tensorboard --logdir logs (<- Folder specified below)
# 4) Go to shown URL: http://localhost:6006

# Access files in WSL Ubuntu using windows explorer:
# https://ling123labs.com/posts/WSL-files-in-Windows-and-vice-versa/
# To access your Linux files in Windows, open the Ubuntu terminal and type: explorer.exe . (<- include the punctuation mark)
# This will open the linux directory in Windows Explorer, with the WSL prefix “\wsl$\Ubuntu-18.04\home\your-username”

import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
import keras
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import os
import glob
import numpy as np
from pathlib import Path

# Prepare training, validation and test dataset
def get_ds(data_dir, batch_size, img_height, img_width, color_mode, label_mode, val_split, test_split, seed, class_names):
    # Training dataset:
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,                               # directory with training images, classes in seperate folders
        labels='inferred',                      # lables are taken from subfolder names
        label_mode=label_mode,                       # OR categorical, binary
        class_names=class_names, 
        color_mode=color_mode,                  # grayscale OR rgb
        batch_size=batch_size,
        image_size=(img_height, img_width),     # images will be reshaped if not in this size
        shuffle=True,
        seed=seed,                              # set seed if training should be the same any time it runs
        validation_split=val_split,             # images used for validation
        subset='training',
    )   
    # Validation dataset:
    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,                              
        labels='inferred',                     
        label_mode=label_mode,  
        class_names=class_names,                      
        color_mode=color_mode,                     
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
    # Split validation dataset further into a validation dataset (at the end of each epoch)
    # and test dataset (at the end of the training)
    # Split batches according to the constant TEST_SPLIT
    ds_test = ds_validation.take(np.floor(int(num_batches)*test_split))
    ds_validation = ds_validation.skip(np.floor(int(num_batches)*test_split))
    return ds_train, ds_validation, ds_test

# Prepare evaluation dataset
def get_pred_ds(pred_dir, batch_size, img_height, img_width, color_mode, label_mode, class_names):
    # Training dataset:
    ds_pred = tf.keras.preprocessing.image_dataset_from_directory(
        pred_dir,                               
        labels='inferred',                      
        label_mode=label_mode, 
        class_names=class_names,                     
        color_mode=color_mode,                
        batch_size=batch_size,
        image_size=(img_height, img_width),     
        shuffle=True,
        seed=None,                             
        validation_split=None,                
        subset=None,
    )  
    return ds_pred

# Normalize images from 8 bit to values between 0-1
def normalize_img(image, label):
    img = tf.cast(image, tf.float32)/255.0
    return img, label

# Function for data augmentation
# Not used for now
def augment_img(image, label):
    # Further data augmentation
    # image = tf.image.random_brightness(image, max_delta=0.1)
    # image = tf.image.random_contrast(image, lower=0.2, upper=0.5)
    image = tf.image.random_flip_left_right(image)  # is done in 50% of the cases
    image = tf.image.random_flip_up_down(image)
    # image = tf.image.per_image_standardization(image)
    return image, label

# Function for data tuning
def tune_img(ds_train, ds_validation, ds_test, cache_ds, cache_on_drive, cache_pth, cache_name):
    # Prepare dataset and configure dataset for performance
    # Setup for training dataset:
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # With caching to memory the program can't train on more than 20.000 images (512x512x3, RTX4090 -> OOM)
    # It can be choosen in the settings that the cached dataset is written to HDD OR RAM
    # See: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    # When caching to a file, the cached data will persist across runs.
    # Even the first iteration through the data will read from the cache file!!!
    # See also: https://www.tensorflow.org/datasets/performances
    if(cache_ds):
        # Cache to hdd or RAM
        if(cache_on_drive):
            ds_train = ds_train.cache(str(cache_pth) + '/' + cache_name)
        else:
            ds_train = ds_train.cache()
    ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Setup for validation dataset:
    ds_validation = ds_validation.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_validation = ds_validation.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Setup for test dataset:
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds_train, ds_validation, ds_test

def tune_pred_img(ds_pred):
    # Prepare prediction dataset and configure it for performance
    ds_pred = ds_pred.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_pred = ds_pred.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds_pred

# Function for callbacks
def get_callbacks(callbacks_enable, checkpoint_path, saving_th):
    # Create empty list for callbacks
    callbacks = []
    
    # Save best weights as checkpoint (highest validation accuracy):
    if(callbacks_enable["save_ckpt"]):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,               # Path to checkpoint file (not only folder)
            save_weights_only=True,                 # False = whole model will be saved
            monitor='val_acc',                 # Value to monitor
            mode='max',                             # max, min, auto fpr value to monitor
            save_best_only=True,                    # save only the best model/weights
            verbose=1,                              # show messages
            save_freq='epoch',                      # check after every epoch
            initial_value_threshold=saving_th,)     # minimum/maximum value for saving
        callbacks.append(model_checkpoint_callback)

    # Early stopping:
    if(callbacks_enable["early_stop"]):
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', 
            patience=20,            # 10-15
            start_from_epoch=40     # 40
        )
        callbacks.append(early_stopping_callback)

    # Tensorboard:
    if(callbacks_enable["tensorboard"]):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir="logs",             # Directory to store log files
            histogram_freq=0,           # frequency (in epochs) at which to compute activation histograms for the layers of the model
            write_graph=True,           # whether to visualize the graph in Tensorboard
            write_images=False,         # whether to write model weights to visualize as image in Tensorboard
            update_freq="epoch",        # 'batch' or 'epoch' or integer.
        )
        callbacks.append(tensorboard_callback)

    # Learning rate scheduler: reduces learning rate dependent on epoch index -> own function!
    if(callbacks_enable["lr_scheduler"]):
        lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(
            lr_scheduler, 
            verbose=1
        )
        callbacks.append(lr_scheduler_callback)

    return callbacks

# Function for reducing the learning rate dependent on the epoch
# For callback 'lr_scheduler_callback'
def lr_scheduler(epoch):
    # SGD optimizer
    learning_rate = 0.01
    if epoch >= 6:
        learning_rate = 0.005
    if epoch >= 12:
        learning_rate = 0.001
    if epoch >= 18:
        learning_rate = 0.0005
    if epoch >= 24:
        learning_rate = 0.0001
    if epoch >= 28:
        learning_rate = 0.00005
    """
    # Old SGD optimizer for grayscale images
    learning_rate = 0.01
    if epoch >= 10:
        learning_rate = 0.005
    if epoch >= 20:
        learning_rate = 0.001
    if epoch >= 30:
        learning_rate = 0.0005
    if epoch >= 40:
        learning_rate = 0.0001
    if epoch >= 45:
        learning_rate = 0.00005
    """
    """ 
    # ADAM optimizer
    learning_rate = 0.00001
    if epoch >= 40:
        learning_rate = 0.000005
    if epoch >= 60:
        learning_rate = 0.000001
    if epoch >= 80:
        learning_rate = 0.0000005
    if epoch >= 90:
        learning_rate = 0.0000001
    """
    return learning_rate

# Function loads a single image for prdictions
# It also normalizes the image
def load_img(pth, category, name, color_mode):
    # Load image
    img = keras.utils.load_img(
        pth / category / name,
        color_mode=color_mode, 
        target_size=None,
    )
    # Convert PIL object to numpy array
    img = keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # Normalize pixel values to 0-1
    img = tf.cast(img, tf.float32)/255.0
    # img = normalize_img()
    return img

# Function predicts a single image and prints output
def predict_single_img(model, data_path, subfolder, img_name, color_mode, class_names, threshold=0.5):
    # Count number of classes
    num_classes = len(class_names)
    # print(class_names)
    # load image
    img = load_img(data_path, subfolder, img_name, color_mode)
    # Predict probabilities: return of a 2D numpy array (why 2D?)
    print(f"Predict class of image \"{img_name}\":")
    probabilities = model.predict(img, verbose=0)
    # Connect probability with the respective class label
    # Binary classification
    if(num_classes == 2):
        if(probabilities <= threshold):
            pred_class_name = class_names[0]
        elif(probabilities > threshold):
            pred_class_name = class_names[1]
        pred_probability = probabilities[0][0]*100 
    # Multi classification
    elif(num_classes > 2):
        class_index = np.argmax(probabilities, axis=-1)
        pred_class_name = class_names[class_index[0]]
        pred_probability = probabilities[0][class_index[0]]*100
    # Print data
    print(f"  -> Image {img_name} belongs to class \"{pred_class_name}\" ", end="")  
    # Binary classification
    if(num_classes == 2): 
        print(f"({pred_probability:.2f}%, 0%={class_names[0]}, 100%={class_names[1]}, thr={(threshold*100):.2f}%)")  
    elif(num_classes > 2):
        print(f"({pred_probability:.2f}%)")

# Function returns a list with all class names
# = names of all subfolders in the data folder
# It also controls what is the negative and what is the positive class for binary classification
# To make ko cells the positive class, the alphabetical order needs to be switched
# https://medium.com/@asimango/the-positive-class-what-should-it-be-in-a-machine-learning-binary-classification-problem-36c316da1127
# https://www.techiedelight.com/list-all-subdirectories-in-directory-python/
def get_class_names(data_dir):
    class_list = []
    for file in os.listdir(data_dir):
        d = os.path.join(data_dir, file)
        # Only folders, no files
        if os.path.isdir(d): 
            class_list.append(file)
    # sort list alphabetically
    num_classes = len(class_list)
    if(num_classes == 2):
        class_list.sort(reverse = True)
    elif(num_classes > 2):
        class_list.sort()
    return class_list

def set_growth_and_print_versions(print_versions=True):
    # Show tensorflow version
    if(print_versions):
        print("\n>> Versions:")
        print("Python: ", sys.version, "")
        print("TensorFlow: ", tf.__version__, "")
        print("Keras: ", keras.__version__, "")
        print("CUDNN: ",tf_build_info.build_info['cudnn_version'])
        print("CUDA: ",tf_build_info.build_info['cuda_version'])
    # Optimize memory
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # List GPUs
    if(print_versions):
        print(">> Available GPUs:")
        for gpu in gpus:
            print(gpu) 

# Function detetes all files in the cache folder
# in order to remove old chached datasets
def clear_ds_cache(cache_pth):
    cache_file = str(cache_pth) + '/*'
    # print(cache_file)
    files = glob.glob(cache_file)
    for f in files:
        os.remove(f)
        # print(f)
    print("\nCached datasets removed!")
    return True

# Function creates all working folders in the root directory of the program
# If they do not exist yet!
def create_prg_folders(data_pth, chkpt_pth, log_pth, plot_pth, vis_pth, pred_pth, cache_pth):
    # https://kodify.net/python/pathlib-path-mkdir-method/
    Path(data_pth).mkdir(parents=True, exist_ok=True)
    Path(chkpt_pth).mkdir(parents=True, exist_ok=True)
    Path(log_pth).mkdir(parents=True, exist_ok=True)
    Path(plot_pth).mkdir(parents=True, exist_ok=True)
    Path(vis_pth).mkdir(parents=True, exist_ok=True)
    Path(pred_pth).mkdir(parents=True, exist_ok=True)
    Path(cache_pth).mkdir(parents=True, exist_ok=True)

# Function returns a confusion matrix displayed in the terminal
# for a specific dataset and a specific trained model
def calc_confusion_matrix(dataset, model, num_classes, print_in_terminal=False, threshold=0.5):
    # Get predictions and labels for the dataset
    # https://stackoverflow.com/questions/64687375/get-labels-from-dataset-when-using-tensorflow-image-dataset-from-directory
    predictions = np.array([])
    labels = np.array([])
    for x, y in dataset:
        # https://stackoverflow.com/questions/59946574/how-to-make-a-prediction-as-binary-output-python-tensorflow
        # Binary classification
        if(num_classes == 2):
            predictions = np.concatenate((predictions, (model.predict(x, verbose=0) > threshold).astype("int32")), axis=None)
        # Multiple classes
        elif(num_classes > 2):
            predictions = np.concatenate([predictions, np.argmax(model.predict(x, verbose=0), axis=-1)])
        labels = np.concatenate((labels, y), axis=0)              
    cm = tf.math.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes).numpy()
    # print(predictions)
    # print(labels)
    if(print_in_terminal):
        print(cm)  
    return cm  

# Function returns true and predicted labels for a specific dataset
def get_labels_and_prediction(dataset, model):
    predict = np.array([])
    labels = np.array([])
    for x, y in dataset:
        labels = np.concatenate((labels, y), axis=0) 
        predict = np.concatenate((predict, model.predict(x, verbose=0)), axis=None)   
    return labels, predict  

# Calculates and returns the parameters for a ROC curve
# tpr = true positive rate, fpr = false positive rate
# Reqires the sklearn library (scikit-learn)! No idea how to do that with tensorflow/keras alone
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
def calc_roc_curve(dataset, model):
    # Get labels and predictions of prediction dataset
    labels, predict = get_labels_and_prediction(dataset, model)
    # Calculate ROC parameters and AUC using sklearn
    fpr, tpr, threshold = roc_curve(labels, predict)
    auc_roc = auc(fpr, tpr)
    # Calculate sweetspot threshold
    # https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    ###########################################################################
    # Variant 1:
    # Calculate the g-mean for each threshold
    # The Geometric Mean or G-Mean is a metric for imbalanced classification that, if optimized, 
    # will seek a balance between the sensitivity and the specificity.
    # G-Mean = sqrt(Sensitivity * Specificity)
    # Sensitivity = TruePositive / (TruePositive + FalseNegative) = True Positive Rate
    # Specificity = TrueNegative / (FalsePositive + TrueNegative) = 1 – False Positive Rate
    gmeans = np.sqrt(tpr * (1 - fpr))
    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    # print(f"Variant 1: index={ix}, threshold={threshold[ix]}")
    ###########################################################################
    # Variant 2: Youden's J statistic
    """
    J = tpr - fpr
    # Locate the index of the largest J value
    ix = np.argmax(J)
    # print(f"Variant 2: index={ix}, threshold={threshold[ix]}")
    """
    ###########################################################################
    # Unlike in the tutorial, the two variants return (slightly) different values?!?
    
    # Return dict
    return {'fpr': fpr, 'tpr': tpr, 'thr_index': ix, 'thr': threshold[ix], 'auc': auc_roc}

# Calculates and returns the parameters for a Precision-Recall-Curve
# tpr = true positive rate, fpr = false positive rate
# Reqires the sklearn library (scikit-learn)! No idea how to do that with tensorflow/keras alone  
# https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
def calc_prec_rec_curve(dataset, model):
    # Get labels and predictions of prediction dataset
    labels, predict = get_labels_and_prediction(dataset, model)  
    # Calculate Precision-Recall_curve parameters
    prec, rec, threshold = precision_recall_curve(labels, predict) 
    auc_prc = auc(rec, prec)
    # Convert to f score
    # F-Measure = (2 * Precision * Recall) / (Precision + Recall)
    fscore = (2 * prec * rec) / (prec + rec) 
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    # Calculate line for random classifier
    random = len(labels[labels==1]) / len(labels)
    # Return dict
    return {'prec': prec, 'rec': rec, 'thr_index': ix, 'thr': threshold[ix], 'rand': random, 'auc': auc_prc}

# Function sets the metrics for training and evaluation
# https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
# https://stackoverflow.com/questions/66635552/keras-assessing-the-roc-auc-of-multiclass-cnn
def set_metrics(num_classes, mode, threshold=0.5):
    # Binary classification
    if(num_classes == 2):
        if(mode == 'train'):
            # Metrics for training (without threashold)
            metrics = [
                keras.metrics.BinaryAccuracy(name='acc'),
                keras.metrics.Precision(name='prec'),
                keras.metrics.Recall(name='rec'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
            ]
        elif(mode == 'eval'):
            # Metrics for evaluation (with threashold)
            metrics = [
                keras.metrics.BinaryAccuracy(threshold=threshold, name='acc'),
                keras.metrics.Precision(name='prec'),
                keras.metrics.Recall(name='rec'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR')
            ]    
        else:
            return False
        
    # Multiclass classification
    elif(num_classes > 2):
        if(mode == 'train'):
            # Metrics for training
            metrics = ["acc"] 
        elif(mode == 'eval'):
            # Metrics for evaluation
            metrics = ["acc"] 
        else:
            return False
        
    return metrics    


"""
# Function returns a dict with all indices and layer names
# of all cnn layers in the model as list
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
"""
"""
# Function counts the number of cnn layers in a model
def num_cnn_layers(model):
    count = 0
    for _, layer in enumerate(model.layers):
        if 'cnn' in layer.name: 
            count += 1
    return count
"""
"""
# Convert RGB image to grayscale
def rgb_to_gray_img(image, label):
    img = tf.image.rgb_to_grayscale(image)
    return img, label
"""

