from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt 
import tensorflow as tf
import seaborn as sns
import numpy as np

tf.config.list_physical_devices('GPU') tf.config.experimental.set_memory_growth(gpus[0], True)

### Preprocessing the Training set
train_datagenerator = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

training_data = train_datagenerator.flow_from_directory('./Data/train',
                                target_size = (64, 64),
                                batch_size = 32,
                                class_mode = 'categorical')


### Preprocessing the Test set
validation_datagenerator = ImageDataGenerator(rescale = 1./255)
validation_data = validation_datagenerator.flow_from_directory('./Data/test',
                                target_size = (64, 64),
                                batch_size = 32,
                                class_mode = 'categorical')


### Initialisation
model = tf.keras.models.Sequential()


### Convolution Layer
model.add(tf.keras.layers.Conv2D(filters = 32,
                                kernel_size = 3,
                                activation = 'relu',
                                input_shape = [64,64,3]))


### Pooling Layer
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


### Activation Functions
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


###  Flattening Layer
model.add(tf.keras.layers.Flatten())


###  Fully Connected Layer
model.add(tf.keras.layers.Dense(units = 132, activation = 'relu'))


###  Output Layer
model.add(tf.keras.layers.Dense(units = 7, activation = 'softmax'))


### Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


### Training the CNN on the Training set and evaluating it on the Validation set
model= model.fit(x = training_data, validation_data = validation_data, epochs = 42)


def plot_model_history(model_history):
    
    #Plotting Accuracy and Loss curves 
    
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()
    
    
model.save('./Models/model.h5')
