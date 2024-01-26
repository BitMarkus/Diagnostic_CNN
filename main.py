import os
# Ignore most messages from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from tensorflow import keras
import pathlib
import matplotlib.pyplot as plt
import random
# Import own classes and functions
from vgg19 import vgg_model
from resnet50 import resnet_model
from xception import xception_model
import fcn
import vis
import menu

# Optimize memory
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# List GPUs
for gpu in gpus:
    print(gpu)
# Show tensorflow version
print("TensorFlow version: ", tf.__version__, "")

# PROGRAM PARAMETERS #
# CLASSES = ['wt', 'ko']
CLASSES = ['WT_1618-02', 'WT_JG', 'WT_KM', 'WT_MS', 'KO_1096-01', 'KO_1618-01', 'KO_BR2986', 'KO_BR3075']
# Path to dataset
DATA_PTH = pathlib.Path('dataset/')
# Path for saved weights
WGHT_PTH = pathlib.Path("weights/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5")
# Path for tensorboard logs
LOG_PTH = pathlib.Path("logs/")
# Path for logging learning rate
LOG_LR_PTH = LOG_PTH / "scalars/learning_rate/"
# Path auto save the plots at the end of the training
PLOT_PTH = pathlib.Path("plots/" + '_'.join(CLASSES) + "/") 
# Path for visualizations
VIS_PTH = pathlib.Path("vis/")
# Path for prediction images
PRED_PTH = pathlib.Path("predictions/")

# IMAGE PARAMETERS #
IMG_HEIGHT = 442
IMG_WIDTH = 550
IMG_CHANNELS = 1    # Image channels -> Grayscale = 1
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
INPUT_SHAPE = (None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# NETWORK HYPERPARAMETERS #
SEED = 620                  # 123
BATCH_SIZE = 32             # 32
VAL_SPLIT = 0.2             # 0.3
NUM_CLASSES = len(CLASSES)
NUM_EPOCHS = 50            # 100
L2_WEIGHT_DECAY = 0         # 0
DROPOUT = 0.5               # 0.5

# CHOOSE MODEL #
MODEL = 'xception'    # OR 'vgg19' OR 'resnet'
OPT_MOMENTUM = 0.9
# Choose optimizer and loss function for network architecture
if(MODEL == 'resnet'):
    LEARNING_RATE = 0.01     # Is also determined in the learning rate scheduler!
    OPT = keras.optimizers.SGD(LEARNING_RATE, OPT_MOMENTUM)
    LOSS = keras.losses.SparseCategoricalCrossentropy(from_logits=False) # from_logits=False: Softmax on output layer
elif(MODEL == 'vgg19'):
    LEARNING_RATE = 0.00001   
    OPT = optimizer=keras.optimizers.Adam(LEARNING_RATE)
    LOSS = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
elif(MODEL == 'xception'):
    LEARNING_RATE = 0.01 
    OPT = keras.optimizers.SGD(LEARNING_RATE, OPT_MOMENTUM)
    LOSS = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Log learning rate in tensorboard
# https://www.tensorflow.org/tensorboard/scalars_and_keras
file_writer = tf.summary.create_file_writer(str(LOG_LR_PTH))
file_writer.set_as_default()

#############
# Main Menu #
#############

while(True):  
    print("\n:MAIN MENU:")
    print("1) Create CNN Network")
    print("2) Show Network Summary")
    print("3) Load Training Data")
    print("4) Train Network")
    print("5) Load Model")
    print("6) Predict random Images in Folder")
    print("7) Predict all Images in Folder")
    print("8) Exit Program")
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
            # Selection of network architecture
            if(MODEL == 'resnet'):
                model = resnet_model(IMG_SHAPE, DROPOUT, NUM_CLASSES)
            elif(MODEL == 'vgg19'):
                model = vgg_model(IMG_SHAPE, DROPOUT, L2_WEIGHT_DECAY, NUM_CLASSES)   
            elif(MODEL == 'xception'):
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
        # Get training, validation and test data
        ds_train, ds_validation, ds_test = fcn.get_ds(DATA_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, VAL_SPLIT, SEED, CLASSES)
        # Get class names
        class_names = ds_train.class_names
        print("Classes: ", class_names)    
        # Data tuning
        AUTOTUNE = tf.data.AUTOTUNE
        ds_train, ds_validation, ds_test = fcn.tune_img(ds_train, ds_validation, ds_test, AUTOTUNE)
        # Data augmentation
        ds_train = ds_train.map(fcn.augment_img, num_parallel_calls=AUTOTUNE)               
        
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
            # Compile model
            model.compile(
                # from_logits=False: Softmax on output layer
                loss=LOSS,
                optimizer=OPT,
                metrics=["accuracy"],
            )

            # Get list with callbacks
            callback_list = fcn.get_callbacks(WGHT_PTH)

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
        # Choose checkpoint
        chkpt = "xcept_SGD_checkpoint-36-0.97_2cl.hdf5"
        # Load checkpoint weights
        print("\n:LOAD MODEL:") 
        if('model' not in globals()):
            print('No network generated yet!')
        else:
            model.load_weights(f"weights/{chkpt}")
            print("Import of weights finished.")
     
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
                        fcn.predict_single_img(model, PRED_PTH, subfolder, pred_img, CLASSES)
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
            ds_pred = fcn.get_pred_ds(PRED_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CLASSES)
            AUTOTUNE = tf.data.AUTOTUNE
            ds_pred = fcn.tune_pred_img(ds_pred, AUTOTUNE)
            # Compile model
            model.compile(
                # from_logits=False: Softmax on output layer
                loss=LOSS,
                optimizer=OPT,
                metrics=["accuracy"],
            )
            print("Evaluate model with prediction dataset:")
            model.evaluate(ds_pred, verbose=1) 

    ################
    # Exit Program #
    ################

    elif(menu1 == 8):
        print("\nExit program...")
        break 
    
    # Wrong Input
    else:
        print("Not a valid option!")      

# Last line in main program to keep plots open after program execution is finished
# see function show_plot_exec() in vis.py
plt.show()
