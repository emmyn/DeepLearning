#
#   The images are organized following kera's needs
#

from keras.models import Sequential        # initialize network
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten           
from keras.layers import Dense             # Add fully connected layers
from keras.layers import Dropout

classifier = Sequential()

# Step Convolution
# Transform picture into matrix and find features
classifier.add(Convolution2D(filters = 32, 
                             kernel_size = 3,
                             strides = 1,
                             input_shape = (150, 150, 3),
                             activation = "relu"))

# Step Max Pooling
# Max value of feature map
classifier.add(MaxPooling2D(pool_size = (2,2),
                         strides = 2))

# Added convolutional layer to increase accuracy
# Added pooling layer because convo goes with a pooling layer
classifier.add(Convolution2D(filters = 32, 
                             kernel_size = 3,
                             strides = 1,
                             activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step Flattening
classifier.add(Flatten())

# Step fully connected layers
classifier.add(Dense(units = 128,
                     activation = "relu"))
classifier.add(Dense(units = 128,
                     activation = "relu"))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, 
                     activation = "sigmoid"))

# Compilation of the network
classifier.compile(optimizer = "adam",
                   loss = "binary_crossentropy",
                   metrics = ["accuracy"])

# Training of the network without overtrainning it 
# Copy Paste from Keras Documentation Image Preprocessing
# with small modifications
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset\\training_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset\\test_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=250,
        epochs=75,
        validation_data=test_set,
        validation_steps=63)


# Evaluating the network
import numpy as np
from keras.preprocessing import image

test_image = image.load_img("dataset/single_prediction/cat_or_dog_2.jpg",
                       target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, 
                            axis = 0)

classifier.predict(test_image)
training_set.class_indices


# Improving the network
# Changed number of epochs, 
#                   size of image
#                   added Dropout
#                   added fully connected layer




