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
print("TensorFlow version: ", tf.__version__, "\n")

# PROGRAM PARAMETERS #
DATA_PTH = pathlib.Path('img_550x442_200/')
CATEGORIES = ['wt', 'ko']
# DATA_PTH = pathlib.Path('wt_test/')
# CATEGORIES = ['wt1', 'wt2']
# DATA_PTH = pathlib.Path('ko_test/')
# CATEGORIES = ['ko1', 'ko2']
# DATA_PTH = pathlib.Path('same_test/')
# CATEGORIES = ['wt11', 'wt12']
# Path for saved weights
CHCKPT_PTH = pathlib.Path("weights/checkpoint-{epoch:02d}-{val_accuracy:.2f}.hdf5")
# Path for tensorboard logs
LOG_PTH = pathlib.Path("logs/")
# Path for logging learning rate
LOG_LR_PTH = LOG_PTH / "scalars/learning_rate/"
# Path auto save the plots at the end of the training
PLOT_PTH = pathlib.Path("plots/" + '_'.join(CATEGORIES) + "/") 

# IMAGE PARAMETERS #
IMG_HEIGHT = 442
IMG_WIDTH = 550
IMG_CHANNELS = 1    # Image channels -> Grayscale = 1
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# NETWORK HYPERPARAMETERS #
SEED = 123                  # 123
BATCH_SIZE = 32             # 32
INPUT_SHAPE = (None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
VAL_SPLIT = 0.3             # 0.3
NUM_CLASSES = 2             # 2
NUM_EPOCHS = 100            # 100
L2_WEIGHT_DECAY = 0
DROPOUT = 0.5               # 0.5
LEARNING_RATE = 0.00001     # Is also determined in the learning rate scheduler

# Log learning rate in tensorboard
# https://www.tensorflow.org/tensorboard/scalars_and_keras
file_writer = tf.summary.create_file_writer(str(LOG_LR_PTH))
file_writer.set_as_default()

# GET TRAINING, VALIDATION, AND TEST DATA #
ds_train, ds_validation, ds_test = fcn.get_ds(DATA_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, VAL_SPLIT, SEED, CATEGORIES)

# CLASS NAMES #
class_names = ds_train.class_names
print("Classes: ", class_names, "\n")

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
train_history = model.fit(
    ds_train, 
    validation_data=ds_validation,
    epochs=NUM_EPOCHS,
    callbacks=callback_list, 
    verbose=1,
)

# LOAD MODEL WEIGHTS #
# model.load_weights("saved_models/checkpoint-55-1.00.hdf5")

# EVALUATE MODEL #
print("\nTest model:")
eval_history = model.evaluate(ds_test, verbose=1)

# PLOT ACCURACY AND LOSS #
fcn.create_metrics_plot(train_history, eval_history, PLOT_PTH, SEED, show_plot=True, save_plot=True)
