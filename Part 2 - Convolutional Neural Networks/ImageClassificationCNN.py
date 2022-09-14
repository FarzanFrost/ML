from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

#Initialising the CNN
classifier = Sequential()

#adding the convolutional layer
#here 32 is the number of feature maps we use
#here 3,3 is the size of the feature map we consider
#input shape parameter considers the shape of the input image,
#64 , 64 is the size of the image array, and 3 is the number of layers
#since the images are color images, we have 3 layers for RGB
#to remove linearity in these feature map we use the activation function
classifier.add( Convolution2D( 32 , 3 , 3 , input_shape = ( 64 , 64 , 3 ) , activation = 'relu' ) )

#adding the max pooling layer
#here using a matrix of 2 x 2 we reduce the size of the feature map
#we take the maximum value here as it is maximum pooling,
#there other type of pooling as well, e.g. - avarage pooling
classifier.add( MaxPooling2D( pool_size = ( 2 , 2 ) ) )

#addition of CNN layers to make this deep learning and to inprove accuracy
classifier.add( Convolution2D( 32 , 3 , 3 , activation = 'relu' ) )
classifier.add( MaxPooling2D( pool_size = ( 2 , 2 ) ) )

#adding the flattening layer
#here it takes all feature maps of the previous layer
#,and flattens them into a single large column vector
classifier.add( Flatten() )

#We add an ANN an the end of the CNN
classifier.add( Dense( units = 128 , activation = 'relu' ) )
classifier.add( Dense( units = 1 , activation = 'sigmoid' ) )

#we compile our CNN
classifier.compile( optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = [ 'accuracy' ] )

#here we rescale the image values between zero and 1
#also we make some transformation to image data randomly
trainDataGen = ImageDataGenerator(
    rescale = 1./255 , 
    shear_range = 0.2 ,
    zoom_range = 0.2 ,
    horizontal_flip = True )

testDataGen = ImageDataGenerator( rescale = 1./255 )

#here first we give the paths to the training data set
#target size represent the size of the image that goes into the CNN
#class mode sets how many types of data we have, e.g. - cats and dogs
trainingSet = trainDataGen.flow_from_directory( 'D:/Practice projects/ML/Part 2 - Convolutional Neural Networks/dataset/training_set' ,
                                                target_size = ( 64 , 64 ) ,
                                                batch_size = 32 ,
                                                class_mode = 'binary' )
testSet = testDataGen.flow_from_directory( 'D:/Practice projects/ML/Part 2 - Convolutional Neural Networks/dataset/test_set' ,
                                                target_size = ( 64 , 64 ) ,
                                                batch_size = 32 ,
                                                class_mode = 'binary' )

classifier.fit_generator( trainingSet ,
                          steps_per_epoch = 8000 ,
                          epochs = 25 ,
                          validation_data = testSet )


testImage1 = image.load_img( 'D:/Practice projects/ML/Part 2 - Convolutional Neural Networks/dataset/single_prediction/cat_or_dog_1.jpg' ,
                           target_size = ( 64 , 64 ) )

testImage2 = image.load_img( 'D:/Practice projects/ML/Part 2 - Convolutional Neural Networks/dataset/single_prediction/cat_or_dog_2.jpg' ,
                           target_size = ( 64 , 64 ) )

testImage1 = image.img_to_array( testImage1 )

testImage2 = image.img_to_array( testImage2 )

testImage1 = np.expand_dims( testImage1 , axis = 0 )

testImage2 = np.expand_dims( testImage2 , axis = 0 )

result1 = classifier.predict( testImage1 )

result2 = classifier.predict( testImage2 )

print( trainingSet.class_indices )
print( "should be a dog " , result1 )
print( "should be a cat " , result2 )
