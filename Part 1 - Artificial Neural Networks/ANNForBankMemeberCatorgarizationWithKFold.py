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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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

def buildClassifier( optimizer ) :
    
    classifier = Sequential()
    classifier.add( ks.Input( shape = ( 11 , ) ) )
    classifier.add( Dense( units = 6 , kernel_initializer = 'uniform' , activation = 'relu' ) )
    classifier.add( Dense( units = 6 , kernel_initializer = 'uniform' , activation = 'relu' ) )
    classifier.add( Dense( units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid' ) )
    classifier.compile( optimizer = optimizer , loss = 'binary_crossentropy' , metrics = [ 'accuracy' ] )
    return classifier

#classifier = KerasClassifier( build_fn = buildClassifier , batch_size = 200 , epochs = 300 )
#accuracies = cross_val_score( estimator = classifier , X = xTrain , y = yTrain , cv = 10 , n_jobs = -1 )

#mean = accuracies.mean()
#variance = accuracies.std()

classifier = KerasClassifier( model = buildClassifier )

parameters = { 'batch_size' : [ 25 , 32 ] ,
               'epochs' : [ 100 , 500 ] ,
               'optimizer' : [ 'adam' , 'rmsprop' ] }
grid_search = GridSearch( estimator = classifier , param_grid = parameters , scoring = 'scoring' , cv = 10 )
grid_search = grid_search.fit( xTrain , yTrain )

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print( "best parameters" , best_parameters )
print( "best accuracy" , best_accuracy )



