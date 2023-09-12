import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Create network model according to:
# https://machinelearningmastery.com/review-of-architectural-innovations-for-convolutional-neural-networks-for-image-classification/
# Similar to VGG16

# Dense = fully connected layers
class FCLayer(layers.Layer):
    def __init__(self, num_nodes, dropout, l2_weight_decay):
        super(FCLayer, self).__init__()
        if(l2_weight_decay == 0):
            self.dense = layers.Dense(
                num_nodes, 
                activation='relu',                        
            )
        else:
            self.dense = layers.Dense(
                num_nodes, 
                activation='relu',                        
                kernel_regularizer=regularizers.l2(l2_weight_decay),
            )
        self.drop = layers.Dropout(dropout)

    def call(self, input_tensor):
        x = self.dense(input_tensor)
        x = self.drop(x)
        return x

# CNN layers
class CNNLayer(layers.Layer):
    def __init__(self, channels, kernel_size=3):
        super(CNNLayer, self).__init__()
        self.conv = layers.Conv2D(
            channels, 
            kernel_size, 
            padding='same'
        )
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        # Common structure in CNNs: BatchNorm -> ReLU
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

# CNN blocks  
class CNNBlock(layers.Layer):
    def __init__(self, num_cnnlayers, channels):
        super(CNNBlock, self).__init__()
        self.num_layers = num_cnnlayers
        # list of cnn layer objects in this block
        self.cnnlayer = [] 
        for i in range(self.num_layers):
            self.cnnlayer.append(CNNLayer(channels))
        self.maxpooling = layers.MaxPooling2D()

    def call(self, input_tensor, training=False):
        x = self.cnnlayer[0](input_tensor, training=training)
        if(self.num_layers > 1):
            for i in range(1, self.num_layers):
                x = self.cnnlayer[i](x, training=training)
        x = self.maxpooling(x)
        return x
  
# CNN model
class CNNModel(keras.Model):
    def __init__(self, shape, dropout, l2_weight_decay=0, num_classes=2):
        super(CNNModel, self).__init__()
        self.shape = shape
        self.block1 = CNNBlock(2, 64)
        self.block2 = CNNBlock(2, 128)
        self.block3 = CNNBlock(3, 256)
        self.block4 = CNNBlock(3, 512)
        self.block5 = CNNBlock(3, 512)
        # self.block6 = CNNBlock(4, 1024)
        self.dense1 = FCLayer(1024, dropout, l2_weight_decay)   # 2048
        self.dense2 = FCLayer(512, dropout, l2_weight_decay)    # 1024
        self.flatten = layers.Flatten()
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, input_tensor, training=False):   
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        # x = self.block6(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classifier(x)
        return x

    # Method to avoid in model summary for output shape "multiple"
    # Overrride model call method
    def model(self):
        x = keras.Input(self.shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

