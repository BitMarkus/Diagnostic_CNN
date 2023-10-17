"""
# GET TRAINING, VALIDATION, AND TEST DATA #
ds_train, ds_validation, ds_test = fcn.get_ds(DATA_PTH, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, VAL_SPLIT, SEED, CATEGORIES)

# CLASS NAMES #
class_names = ds_train.class_names
print("Classes: ", class_names, "\n")

# VIZUALIZE DATA #
# vis.plot_img_batch(BATCH_SIZE, ds_train, VIS_PTH, show_plot=True, save_plot=False)

# DATA TUNING # 
AUTOTUNE = tf.data.AUTOTUNE
ds_train, ds_validation, ds_test = fcn.tune_img(ds_train, ds_validation, ds_test, AUTOTUNE)

# DATA AUGMENTATION #
ds_train = ds_train.map(fcn.augment_img, num_parallel_calls=AUTOTUNE)

# CALLBACKS #
callback_list = fcn.get_callbacks(WGHT_PTH)

# CREATE MODEL #
# Create model using subclassing
model = cnn.CNNModel(IMG_SHAPE, DROPOUT, L2_WEIGHT_DECAY, NUM_CLASSES)
# Build model and print summary
model.build(INPUT_SHAPE)
# A normal summary call does not display output shapes (only 'multiple'):
print(model.model().summary())

# Create model from function
model = cnn_model(IMG_SHAPE, DROPOUT, L2_WEIGHT_DECAY, NUM_CLASSES)
# Print summary of the model
print(model.summary())

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
model.load_weights("weights/checkpoint-52-1.00.hdf5")

# EVALUATE MODEL #
# print("\nTest model:")
# eval_history = model.evaluate(ds_test, verbose=1)

# VISUALIZE #
# Show first x filters of a cnn layer
# vis.plot_filters_of_layers(model, 10)

# Load image
subfolder = 'wt'
img_name = 'WT_01_m555_ORG.png'
# Plot feature maps
vis.plot_image(DATA_PTH, VIS_PTH, subfolder, img_name, save_plot=True, show_plot=True)
vis.plot_feature_maps_of_multiple_layers(model, DATA_PTH, VIS_PTH, subfolder, img_name, num_rows=3, num_cols=3, save_plot=True, show_plot=False)

# PREDICT SINGLE IMAGE #
fcn.predict_single_img(model, DATA_PTH, subfolder, img_name, class_names)

# PLOT ACCURACY AND LOSS #
# vis.plot_metric(train_history, eval_history, PLOT_PTH, SEED, show_plot=True, save_plot=True)
"""