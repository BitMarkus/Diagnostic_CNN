import os
# Ignore most messages from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import keras
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
# Import own classes and functions
from xception import xception_model
import fcn
import vis
import menu

# Set memory growth and print program versions
fcn.set_growth_and_print_versions(print_versions=True)

#########################
# DELETE CACHE FILE !!! #
#########################

# IMAGE PARAMETERS #
IMG_HEIGHT = 512
IMG_WIDTH = 512 
COLOR_MODE = 'rgb'  # Or 'grayscale'
# Number of channels depending on color mode
if(COLOR_MODE == 'rgb'):
    IMG_CHANNELS = 3
elif(COLOR_MODE == 'grayscale'):
    IMG_CHANNELS = 1
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
INPUT_SHAPE = (None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# CALLBACK PARAMETER #
# Enable or disable callbacks
CALLBACKS_ENABLE = {"save_ckpt": True, "early_stop": False, "tensorboard": True, "lr_scheduler": True}
# Minimum validation accuracy from which on checkpoints will be saved
SAVING_THRESHOLD = 0.8

# NETWORK HYPERPARAMETERS FOR XCEPTION NETWORK #
SEED = 111                  # 123
BATCH_SIZE = 32             # max 32 for 512x512px grayscale or rgb images
VAL_SPLIT = 0.2            # 0.2
NUM_EPOCHS = 15             # 50; with a lot of training images (> 10,000), even 10 epochs are enough
OPT_MOMENTUM = 0.9          # 0.9
# Learning rate:
# If the lr callback isn't disabled, the lr is determined by the learning rate scheduler!
# Or else the following value is used during the whole training:
LEARNING_RATE = 0.01        # 0.01 for SGD, 0.00001 for ADAM     
OPT = keras.optimizers.SGD(LEARNING_RATE, OPT_MOMENTUM)
LOSS = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# L2_WEIGHT_DECAY = 0       -> Currently not in use
# DROPOUT = 0.5             -> Currently not in use 

# PROGRAM PARAMETERS #
# Path to dataset
DATA_PTH = pathlib.Path('dataset/')
# Path for checkpoints folder
CHKPT_PTH = pathlib.Path('chkpt/')
# Path for saving checkpoints including file name
CHKPT_FILE_PTH = pathlib.Path(CHKPT_PTH / "checkpoint-{epoch:02d}-{val_accuracy:.2f}.weights.h5")
# Path for tensorboard logs
LOG_PTH = pathlib.Path("logs/")
# Path auto save the plots at the end of the training
PLOT_PTH = pathlib.Path("plots/") 
# Path for visualizations
VIS_PTH = pathlib.Path("vis/")
# Path for prediction images
PRED_PTH = pathlib.Path("predictions/")
# Path to cached datasets
CACHE_PTH = pathlib.Path("cache/")
# Name for cached dataset
CACHE_NAME = "ds.cache"
# File extension for saved checkpoints
CHKPT_EXT = ".weights.h5"

# CLASS PARAMETERS #
# Class names according to the subfolder structure in the data (and prediction) folder 
CLASS_NAMES = fcn.get_class_names(DATA_PTH)
NUM_CLASSES = len(CLASS_NAMES)

# CACHE PARAMETERS #
# Parameter to determine if training dataset is suppose to be cached
# If the dataset is big, caching needs to be done to the HDD or else you ran out of RAM
# Caching the dataset makes training faster, but it requires a lot of hard disk space or RAM
CACHE_DS = True
# Parameter determines if data is cached to memory or hard disk drive
# True: data is cached as a file on the hard disk drive, False: cached to RAM
CACHE_ON_DRIVE = True
# Parameter to clear cached old datasets from the cache/ folder
# If the same dataset is trained as before, set it to False
# When the dataset is changed, set it to True
# Only of importance, if CACHE_DS and CACHE_ON_DRIVE is set to True
CLEAR_CACHE = True

#############
# Main Menu #
#############

while(True):  
    print("\n:MAIN MENU:")
    print("1) Create CNN Network")
    print("2) Show Network Summary")
    print("3) Load Training Data")
    print("4) Train Network")
    print("5) Load Checkpoint")
    print("6) Predict random Images in Folder")
    print("7) Predict all Images in Folder")
    print("8) Plot confusion matrix")
    print("9) Exit Program")
    menu1 = int(menu.input_int("Please choose: "))

    ######################
    # Create CNN Network # 
    ###################### 

    if(menu1 == 1):       
        print("\n:NEW CNN NETWORK:")  
        # Check if a model is already existing
        if('model' in globals()):
            print("A network already exists!")
        else:
            print("Creating new network...")
            model = xception_model(IMG_SHAPE, NUM_CLASSES)       
            print("New network finished.")

    ########################
    # Show Network Summary #  
    ########################

    elif(menu1 == 2):       
        print("\n:SHOW NETWORK SUMMARY:")   
        if('model' in globals()):
            print(model.summary())     
        else:
            print("No network generated yet!") 
        
    ######################
    # Load Training Data # 
    ######################
     
    elif(menu1 == 3):       
        print("\n:LOAD TRAINING DATA:")  
        print("Classes: ", CLASS_NAMES)   
        # Get training, validation and test data
        ds_train, ds_validation, ds_test = fcn.get_ds(DATA_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, COLOR_MODE, VAL_SPLIT, SEED, CLASS_NAMES) 
        # Data tuning
        ds_train, ds_validation, ds_test = fcn.tune_img(ds_train, ds_validation, ds_test, CACHE_DS, CACHE_ON_DRIVE, CACHE_PTH, CACHE_NAME)
        # Data augmentation -> Flipping the image is not helpful because of the orientation of the DIC images
        # ds_train = ds_train.map(fcn.augment_img, num_parallel_calls=tf.data.AUTOTUNE)               
        
    #################
    # Train Network #  
    #################
              
    elif(menu1 == 4):
        print("\n:TRAIN NETWORK:", end='') 
        # https://stackoverflow.com/questions/21980874/how-do-i-check-if-both-of-two-variables-exists-in-python
        if('model' not in globals()):
            print('No CNN generated yet!')
        elif('ds_train' not in globals()):
            print('No training data loaded yet!')
        else:
            # Clear cache folder from old cached datasets
            if(CACHE_DS and CACHE_ON_DRIVE and CLEAR_CACHE):
                fcn.clear_ds_cache(CACHE_PTH)

            # Compile model
            model.compile(
                # from_logits=False: Softmax on output layer
                loss=LOSS,
                optimizer=OPT,
                metrics=["accuracy"],
            )

            # Get list with callbacks
            callback_list = fcn.get_callbacks(CALLBACKS_ENABLE, CHKPT_FILE_PTH, SAVING_THRESHOLD)

            # Train model
            train_history = model.fit(
                ds_train, 
                validation_data=ds_validation,
                epochs=NUM_EPOCHS,
                callbacks=callback_list, 
                verbose=1,
            )  
            print("Training of network finished.\n")  
            # Evaluate model
            print("Evaluate model with test dataset:")
            eval_history = model.evaluate(ds_test, verbose=1) 
            vis.plot_metrics(train_history, eval_history, PLOT_PTH, SEED, show_plot=True, save_plot=True)  
        
    ##############
    # Load Model #
    ##############

    elif(menu1 == 5):
        # Load checkpoint weights
        print("\n:LOAD CHECKPOINT:") 
        if('model' not in globals()):
            print('No network generated yet!')
        else:
            # Choose checkpoint
            inp = str(menu.input_empty("Enter checkpoint name (without extension): "))
            chkpt_import_path = CHKPT_PTH / (str(inp) + str(CHKPT_EXT))
            # print(chkpt_import_path)
            # Check if checkpoint file is in the weights/ folder
            if(os.path.exists(chkpt_import_path)):
                model.load_weights(chkpt_import_path)
                print("Import of weights finished.")
            else:
                print(f"Checkpoint file {chkpt_import_path} not found!") 

    #####################################
    # Predict random images in a folder #
    #####################################

    elif(menu1 == 6):
        print("\n:PREDICT RANDOM IMAGES IN FOLDER:") 
        if('model' not in globals()):
            print('No network generated yet!')
        else:
            # Subfolder for prediction
            subfolder = menu.input_empty('Enter a folder name: ')
            # Number of images for prediction in the folder "predictions"
            num_img = menu.input_int('Enter number of images to predict: ')
            img_pth = PRED_PTH / subfolder
            # Check if the folder exists within the predict folder
            if(os.path.isdir(img_pth)): 
                # Get a list of all files in the specified folder
                folder_list = [f for f in listdir(img_pth) if isfile(join(img_pth, f))]
                # Check if there are more images in the folder than images to predict
                if(num_img <= len(folder_list)):
                    # Choose x random images from the list
                    # https://pynative.com/python-random-choice/
                    choice_list = random.choices(folder_list, k=num_img)
                    # print(choice_list)
                    # Make predictions
                    for pred_img in choice_list:
                        fcn.predict_single_img(model, PRED_PTH, subfolder, pred_img, COLOR_MODE, CLASS_NAMES)
                else:
                    print(f"Not enough images in folder (max {len(folder_list)})!")     
            else:
                print("Folder does not exist!")   

    ############################
    # Predict images in folder #  
    ############################

    elif(menu1 == 7):  
        print("\n:PREDICT IMAGES IN FOLDER:") 
        if('model' not in globals()):
            print('No network generated yet!')
        else:
            # Get dataset for prediction
            ds_pred = fcn.get_pred_ds(PRED_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, COLOR_MODE, CLASS_NAMES)
            ds_pred = fcn.tune_pred_img(ds_pred)
            # Compile model
            model.compile(
                # from_logits=False: Softmax on output layer
                loss=LOSS,
                optimizer=OPT,
                metrics=["accuracy"],
            )
            print("Evaluate model with prediction dataset:")
            model.evaluate(ds_pred, verbose=1) 

    #########################
    # Plot confusion matrix #
    #########################
            
    elif(menu1 == 8):
        print("\n:PLOT CONFUSION MATRIX:") 
        if('model' not in globals()):
            print('No network generated yet!')
        else:
            # Check folders in prediction folder
            pred_folders = fcn.get_class_names(PRED_PTH)
            if(pred_folders != CLASS_NAMES):
                print('Folder structure in predict/ folder does not match with dataset/ folder!')
            else:
                # Get dataset for prediction
                ds_pred = fcn.get_pred_ds(PRED_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, COLOR_MODE, CLASS_NAMES)
                ds_pred = fcn.tune_pred_img(ds_pred)
                # Get predictions and labels for the test dataset
                # https://stackoverflow.com/questions/64687375/get-labels-from-dataset-when-using-tensorflow-image-dataset-from-directory
                predictions = np.array([])
                for x, y in ds_pred:
                    predictions = np.concatenate([predictions, np.argmax(model.predict(x, verbose=0), axis=-1)])
                labels = np.concatenate([y for x, y in ds_pred], axis=0)
                # Create confusion matrix and normalizes it over predicted (columns)
                # Make a numpy array of the matrix data
                cm_result = tf.math.confusion_matrix(labels=labels, predictions=predictions, num_classes=NUM_CLASSES).numpy()
                vis.plot_confusion_matrix(cm_result, CLASS_NAMES, PLOT_PTH, show_plot=True, save_plot=True)             

    ################
    # Exit Program #
    ################

    elif(menu1 == 9):
        print("\nExit program...")
        break
    
    # Wrong Input
    else:
        print("Not a valid option!")      

# Last line in main program to keep plots open after program execution is finished
# see function show_plot_exec() in vis.py
plt.show()
