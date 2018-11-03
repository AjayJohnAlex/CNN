# Part 1 of CNN

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten


#Initialise the CNN 
classifier = Sequential()
#Adding the Layers
#Adding Convolutional layer
# We need mention the no of feature detector dimension as first argument
# and secondly the shape of the feature detector as second argument 
# then we need to mention into which typr of array our images would be converted 
#: first the dimensions of array and no of channels
# then we add the activattion fn to remove the non linearity 
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
# we need to add a second convolutional layer to increase the accuracy
# Adding the Pooling to calculate the pooled feature map
#this is done to reduce the size in feature map so that they
# can be given to be flattened and thus reducing the no of nodes of the fully connected layers
# the pool_size is the  factors by which to downscale 
#pooling reduces the complexity of the model without reducing the performannce 
classifier.add(MaxPooling2D(pool_size=(2,2)))
#we add a second convolutional layer to increase to accuracy of our output
#but here we are not going to use the input shape of an image but the input shape of the max pooled layer
#we can add 1 more convo2D layer with 64 features to get optimal results
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#flattening is done to put all features into a single vector and is used as input to our ANN
#we dont use the spatial structure of the image as they are preserved in the convolution step
#step 3 : Flattening
classifier.add(Flatten())
#step 4 : fully connected ANN
#use the flattened vector as the input
classifier.add(Dense(activation ="relu",units = 128))
#output layer
classifier.add(Dense(activation ="sigmoid",units = 1))
# now we compile the CNN
classifier.compile(optimizer= 'adam',loss = 'binary_crossentropy',metrics= ['accuracy'])
# now fit the CNN to our images
# we will do image augmentation that consist preprocessing of image sto prevent from over fitting
#i.e : great result in training set but not in test set that is over fitting
#image augmentation creates many batches of images and each batch it would apply  
# random transformation by rotating ,flipping them and eventually we get many more diverse images
#it enriches our dataset without adding more images
# we need a class to use image augmentation
from keras.preprocessing.image import ImageDataGenerator
#Creating ImageGenerator 
#Refer Keras Documentation on ImageDataGenerator to get a better idea about Image Augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,#all our pixels btwn 0 and 1
        shear_range=0.2,# random transformation
        zoom_range=0.2,# how much random transformation
        horizontal_flip=True)#flipping the image

test_datagent = ImageDataGenerator(rescale=1./255)
#creating training training data 
training_set = train_datagen.flow_from_directory(
        'data/train',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
#creating tetsing data
test_set = train_datagen.flow_from_directory(
        'data/validation',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
#fitting our CNN model to the images
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,# no of training images
        epochs=25,# no of loops 
        validation_data=test_set,
        validation_steps=2000)# no of testing images