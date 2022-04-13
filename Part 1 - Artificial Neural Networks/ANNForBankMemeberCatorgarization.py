import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras as ks
from keras.models import Sequential #required to initialize our neural network
from keras.layers import Dense #required to build the layers of ANN
from sklearn.metrics import confusion_matrix

#importing data 
#Selecting only columns that will effect the customers to leave
#and seperating dependant and independant variables
dataSet = pd.read_csv( 'Churn_Modelling.csv' )
x = dataSet.iloc[ : , 3 : 13 ].values
y = dataSet.iloc[ : , 13 ].values

#encoding categorical data
#As the model cant understand categorical data, we encode them into numbers
labelEncoderXForGeography = LabelEncoder()
x[ : , 1 ] = labelEncoderXForGeography.fit_transform( x[ : , 1 ] )
labelEncoderXForGender = LabelEncoder()
x[ : , 2 ] = labelEncoderXForGender.fit_transform( x[ : , 2 ] )

#after encoding, the numbers are in the same column,
#so the model will assume there might be a relationship
#between the values, therefore we have to separete them into separate columns
#the categorical_features = [ 1 ], represents the index that needs to be encoded
oneHotEncoder = ColumnTransformer( [ ( "oneHotEncoder" , OneHotEncoder() , [ 1 ] ) ] , remainder = "passthrough" )
x = oneHotEncoder.fit_transform( x )

#oneHotEncoder has produced 3 sepereta columns for Geography
#And these are called dummy variables
#We have to remove one dummy variable, 
#to avoid falling into dummy varaible trap
#after the next line there is no dummy variable trap
x = x[ : , 1 : ]

#we have to separate date to train and test
xTrain , xTest , yTrain , yTest = train_test_split( x , y , test_size = 0.2 , random_state = 0 )

#next we need to scale our data, this is called feature scaling

standardScaler = StandardScaler()
xTrain = standardScaler.fit_transform( xTrain )
xTest = standardScaler.fit_transform( xTest )
#print( "xTrain" , xTrain[ : 5 , : ] )
#print( "xTest" , xTest[ : 5 , : ] )

#Making the ANN
#Here we initailize our classifier as an ANN, which is done by the sequential()
classifier = Sequential()

#adding the input and the first hidden layer
#output_dim is the number nodes in the hidden layer
#which is the average of number of input nodes (11) and number of output nodes (1) of the ANN
#init, initialises the matrix with random numbers close to zero, which is done using uniform
#it is a uniform distribution
#activation determines the activation function here 'relu' refers to the rectifier function
#here input_dim is the number of nodes in the input layer
classifier.add( ks.Input( shape = ( 11 , ) ) )
classifier.add( Dense( units = 6 , kernel_initializer = 'uniform' , activation = 'relu' ) )

#second hidded layer
#here the input_dim is not given, since there is a layer already added the next leyer 
#-knows what to expect.
#here the init works for the weights that coming into this layer
classifier.add( Dense( units = 6 , kernel_initializer = 'uniform' , activation = 'relu' ) )

#final layer
classifier.add( Dense( units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid' ) )

#compile the ANN
#optimizer is for find the optimal set of weights, so we use adamn algorithm, its stocastic gradient descent algorithm
#loss is define the loss function, its a logarithmic function based on the maths behind sigmoid functions and relavent errors
#if the variable has a binary outcome then this logarithmic funciton is 'binary_crossentrophy'
#if the variable has more than 2 outcomes, like 3 categories then this logarithmic funciton is 'cotegorical_crossentrophy'
#since we choose the accuracy as metrics, our ANN will improve on its accuracy little by little on each epoch
classifier.compile( optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = [ 'accuracy' ] )

#fitting ANN to training set
#batch_size - the ann runs the data for a single batch then adjusts the weights
classifier.fit( xTrain , yTrain , batch_size = 200 , epochs = 300 )

#predicting the test set results
#here we query the ANN for all the test data, which returns an array of data.
yPredicted = classifier.predict( xTest )

#Here we convert the data to true or false list, so that the confusion matrix can work on it.
#RHS we have if result is gt 0.5 then true else false
yPredicted =  ( yPredicted > 0.5 )


cm = confusion_matrix( yTest , yPredicted )
print()
print( cm )
print( 'Accuracy' , ( cm[ 0 , 0] + cm[ 1 , 1] ) / 2000 * 100 , ' %' )

