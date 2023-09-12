import os
# Ignore most messages from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import pathlib
# Import own classes and functions
import cnn
import fcn

# Experimental:
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(tf.__version__)

# PROGRAM PARAMETERS #
DATA_DIR = pathlib.Path('550x442_300_autocontr/')
# DATA_DIR = pathlib.Path('275x221/')
CHCKPT_PTH = pathlib.Path("saved_weights/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5")
SEED = 555      # 123

# IMAGE PARAMETERS #
# Small images:
# IMG_HEIGHT = 221
# IMG_WIDTH = 275
# Large images:
IMG_HEIGHT = 442
IMG_WIDTH = 550
# Image channels -> Grayscale = 1
IMG_CHANNELS = 1    
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# NETWORK HYPERPARAMETERS #
BATCH_SIZE = 32             # 32
INPUT_SHAPE = (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# INPUT_SHAPE = (None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) <- Or this?
VAL_SPLIT = 0.3             # 0.3
NUM_CLASSES = 2             # 2
NUM_EPOCHS = 100
L2_WEIGHT_DECAY = 0
DROPOUT = 0.5               # 0.5
LEARNING_RATE = 0.00001    # 0.000005-0.00001 

# GET TRAINING, VALIDATION, AND TEST DATA #
ds_train, ds_validation, ds_test = fcn.get_ds(DATA_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, VAL_SPLIT, SEED)

# CLASS NAMES #
class_names = ds_train.class_names
print("Classes: ", end='')
print(class_names)

# VIZUALIZE DATA #
# fcn.print_img_batch(BATCH_SIZE, ds_train, class_names)

# DATA TUNING # 
AUTOTUNE = tf.data.AUTOTUNE
ds_train, ds_validation, ds_test = fcn.tune_img(ds_train, ds_validation, ds_test, AUTOTUNE)

# DATA AUGMENTATION #
ds_train = ds_train.map(fcn.augment_img, num_parallel_calls=AUTOTUNE)

# VIZUALIZE DATA #
# fcn.print_img_batch(BATCH_SIZE, ds_train, class_names)

# CALLBACKS #
callback_list = fcn.callbacks(CHCKPT_PTH)

# CREATE MODEL #
model = cnn.CNNModel(IMG_SHAPE, DROPOUT, L2_WEIGHT_DECAY, NUM_CLASSES)
# Build model and print summary
model.build(INPUT_SHAPE)
# A normal summary call does not display output shapes (only 'multiple'):
print(model.model().summary())
print()

# COMPILE MODEL #
model.compile(
    # from_logits=False: Softmax on output layer
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    metrics=["accuracy"],
)

# TRAIN MODEL #
print("Train model:")
history = model.fit(
    ds_train, 
    validation_data=ds_validation,
    epochs=NUM_EPOCHS,
    callbacks=callback_list, 
    verbose=1,
)

# LOAD MODEL WEIGHTS #
# model.load_weights("saved_models/checkpoint-55-1.00.hdf5")  # checkpoint-39-0.99.hdf5 checkpoint-55-1.00.hdf5

# EVALUATE MODEL #
print("\nTest model:")
model.evaluate(ds_test, verbose=1)

# PLOT ACCURACY AND LOSS #
fcn.print_acc_loss(history)
