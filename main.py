import os
# Ignore most messages from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import pathlib
import matplotlib.pyplot as plt
# Import own classes and functions
from cnn import cnn_model
import fcn
import vis
import menu

# Optimize memory and show tensorflow version
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("TensorFlow version: ", tf.__version__, "")

# PROGRAM PARAMETERS #
CATEGORIES = ['wt', 'ko']
# Path to dataset
DATA_PTH = pathlib.Path('dataset/')
# Path for saved weights
WGHT_PTH = pathlib.Path("weights/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5")
# Path for tensorboard logs
LOG_PTH = pathlib.Path("logs/")
# Path for logging learning rate
LOG_LR_PTH = LOG_PTH / "scalars/learning_rate/"
# Path auto save the plots at the end of the training
PLOT_PTH = pathlib.Path("plots/" + '_'.join(CATEGORIES) + "/") 
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
SEED = 442                  # 123
BATCH_SIZE = 32             # 32
VAL_SPLIT = 0.3             # 0.3
NUM_CLASSES = 2             # 2
NUM_EPOCHS = 80             # 100
L2_WEIGHT_DECAY = 0         # 0
DROPOUT = 0.5               # 0.5
LEARNING_RATE = 0.00001     # Is also determined in the learning rate scheduler

# Log learning rate in tensorboard
# https://www.tensorflow.org/tensorboard/scalars_and_keras
file_writer = tf.summary.create_file_writer(str(LOG_LR_PTH))
file_writer.set_as_default()

# Get list with callbacks
callback_list = fcn.get_callbacks(WGHT_PTH)

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
    print("6) Predict Single Image")
    print("7) Predict Images in Folder")
    print("8) Exit Program")
    menu1 = int(menu.input_int("Please choose: "))

    ##### Create CNN Network #####  
    if(menu1 == 1):       
        print("\n:NEW CNN NETWORK:")  
        # Check if a model is already existing
        if('model' in globals()):
            print("A network already exists!")
        else:
            print("Creating new network...")
            model = cnn_model(IMG_SHAPE, DROPOUT, L2_WEIGHT_DECAY, NUM_CLASSES)
            print("New network finished.")

    ##### Show Network Summary #####  
    elif(menu1 == 2):       
        print("\n:SHOW NETWORK SUMMARY:")   
        if('model' in globals()):
            print(model.summary())     
        else:
            print("No network generated yet!") 
        
    ##### Load Training Data #####  
    elif(menu1 == 3):       
        print("\n:LOAD TRAINING DATA:")  
        # Get training, validation and test data
        ds_train, ds_validation, ds_test = fcn.get_ds(DATA_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, VAL_SPLIT, SEED, CATEGORIES)
        # Get class names
        class_names = ds_train.class_names
        print("Classes: ", class_names)    
        # Data tuning
        AUTOTUNE = tf.data.AUTOTUNE
        ds_train, ds_validation, ds_test = fcn.tune_img(ds_train, ds_validation, ds_test, AUTOTUNE)
        # Data augmentation
        ds_train = ds_train.map(fcn.augment_img, num_parallel_calls=AUTOTUNE)               
        
    #####Train Network #####            
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
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=keras.optimizers.Adam(LEARNING_RATE),
                metrics=["accuracy"],
            )
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
        
    ##### Load Model #####
    elif(menu1 == 5):
        # Choose checkpoint
        chkpt = "checkpoint-67-0.99_ds1.hdf5"
        # Load checkpoint weights
        print("\n:LOAD MODEL:") 
        if('model' not in globals()):
            print('No network generated yet!')
        else:
            model.load_weights(f"weights/{chkpt}")
            print("Import of weights finished.")
     
    ##### Predict single image #####
    elif(menu1 == 6):
        # Load image
        subfolder = 'wt'
        img_name = 'xxx.png'
        # Make prediction
        print("\n:PREDICT SINGLE IMAGE:") 
        if('model' not in globals()):
            print('No network generated yet!')
        else:
            class_names = fcn.get_class_names(DATA_PTH)
            fcn.predict_single_img(model, DATA_PTH, subfolder, img_name, class_names)

    ##### Predict images in folder #####  
    elif(menu1 == 7):  
        print("\n:PREDICT IMAGES IN FOLDER:") 
        if('model' not in globals()):
            print('No network generated yet!')
        else:
            ds_pred = fcn.get_pred_ds(PRED_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, CATEGORIES)
            AUTOTUNE = tf.data.AUTOTUNE
            ds_pred = fcn.tune_pred_img(ds_pred, AUTOTUNE)
            # Compile model
            model.compile(
                # from_logits=False: Softmax on output layer
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=keras.optimizers.Adam(LEARNING_RATE),
                metrics=["accuracy"],
            )
            print("Evaluate model with prediction dataset:")
            model.evaluate(ds_pred, verbose=1) 

    ##### Exit Program #####
    elif(menu1 == 8):
        print("\nExit program...")
        break 
    
    ##### Wrong Input #####  
    else:
        print("Not a valid option!")      

# Last line in main program to keep plots open after program execution is finished
# see function show_plot_exec() im vis.py
plt.show()
