# Diagnostic_CNN

The aim of this project is to diagnose diseases based on fibroblast cultures of patients. 

The microscopic images of the cells consists of three channels:
- Brightfield image (DIC contrast)
- Lysosomal staining (Lysotracker green)
- Cell membrane staining (WGA deep red)

Training images are taken on a Zeiss microscope with a 63x oil objective. With the help of a motorized stage, mosaic images are created (x=10, y=13), which are then automatically sliced to the format of the training images (512x512x3). The Python script for automatic slicing can be found here: https://github.com/BitMarkus/Read_CZI. The finished training images are saved as 8-bit RGB images in .png format.

After testing several different CNN architectures (VGG, DensNet, ResNet), it has become clear that the Xception network appears to be the most suitable for these purposes. Due to data protection, no training images are included. This is not yet a fully-fledged program. The project is only just beginning, which is why the source code is constantly being adapted and improved.

After cloning the repository the following folders need to be created in the program folder:
- dataset/: This is the folder for training images. For each class a subfolder must be created in dataset/. The name of the class folders are taken as class labels.
- logs/: In case Tensorboard is used for monitoring training progress.
- plots/: Folder for plots. E.g. training/validation accuracy and loss as well as the learning rate are saved automatically as a plot at the end of a training.
- predictions/: This is the folder for images to predict. It should have the same subfolders as the dataset/ folder. Using the images in this folder, a confusion matrix can be generated.
- vis/: Currently not used. This folder was for saving vizualized cnn filters and feature maps. The source code for these visualizations can be found in the file vis.py, but it is currently not used.
- chkpt/: Here, checkpoints are stored during training. Checkpoint saving starts, when validatin accuracy is over 80% (can be adjusted). Then, only checkpoints with a higher accuracy than already saved ones will be saved.
- cache/: When training on a lot of images, the cached dataset might not fit into memory. That's why cached data is stored in the cache/ folder. When training on a new dataset, the constant CLEAR_CACHE must be set to True. In this case, old cashed data is automatically deleted from the cache/ folder before the new training starts. Cachin the training dataset is optional (constant CACH_DS True or False) and will increase training time.

The folder models/ is not relevant as it contains the source code for implementing DensNet, ResNet and VGG networks. It is just for testing other CNN architectures.

The program has a menu with the following options:
- Create CNN Network: Creates a Xception network taking an input tensor of 512x512x3 and randomly initiates weights and biases
- Show Network Summary
- Load Training Data: Three groups will be generated. 1) Training images: for training the network (70% of all images, validation split can be configured). 2) Validation images: To check training progress after each epoch (10%). 3) Test images: to check training success after training is finished.
- Train network: In order to do that, a network needs to be initialized and training images nedd to be loaded first.
- Load checkpoint: Loads a saved checkpoint
- Predict random Images in Folder: A subfolder in the prediction/ folder needs to be specified and the amount of images to predict. Then random images will be choosen and the prediction result will be displayed.
- Predict all Images in Folder: Predicts all images in a folder
- Plot confusion matrix: After training or loading a checkpoint, all images in the folder prediction/ will be predicted an a confusion matrix will be plotted.

All program parameters are listed at the beginning of the program and can be adjusted to the specific needs. The most important ones are: image height and width in px, color mode, validation seed, batch size, validation split, and number of epochs. The learning rate is determined by a learning rate scheduler and can be found under fcn.py, function: lr_scheduler(). L2 weight decay and dropout are obsolete as global average pooling is used before the classifier instead of dense layers. Stochastic gradient descent (SGD) is used as optimizer with a momentum of 0.9. For this project it works better than ADAM.

Tested under the following environment:
Python:  3.11.7
TensorFlow:  2.15.0 
Keras:  3.0.5 
CUDNN:  8
CUDA:  12.2
NVIDIA RTX 4090 or NVIDIA RTX A5000 (24 GB VRAM)