import numpy as np
import scipy.special as sc #sigmoid function is called expit() in tihs library
import matplotlib.pyplot as plt
import time
%matplotlib inline

class neuralNetwork:
    def __init__( self , inputNodes , hiddenNodes , outputNodes , learningRate ) :
        
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        self.learningRate = learningRate
        #the weights are preferd to be in range of -a to +a of normal distribution, where a = 1/sqrt( number of incoming links to a node )
        self.weightInputHidden = np.random.normal( 0.0 , pow( self.inputNodes , -0.5 ) , ( self.hiddenNodes , self.inputNodes ) )
        
        self.weightHiddenOutput = np.random.normal( 0.0 , pow( self.hiddenNodes , -0.5 ) , ( self.outputNodes , self.hiddenNodes ) )
        
        self.activationFunction = lambda x : sc.expit( x )
        
        #inverse of the sigmoid function to calculate the backquery
        self.inverseActivationFunction = lambda x : sc.logit( x )
    
    def train( self , input_list , target_list ) :
        
        inputs = np.array( input_list , ndmin = 2 ).T
        targets = np.array( target_list , ndmin = 2 ).T
        
        #inputs of hidden layer
        hidden_inputs = np.dot( self.weightInputHidden , inputs )
        
        #outputs of hidden layer
        hidden_outputs = self.activationFunction( hidden_inputs )
        
        #input of output layer
        final_inputs = np.dot( self.weightHiddenOutput , hidden_outputs )
        
        #output of the hidden layer
        final_outputs = self.activationFunction( final_inputs )
        
        #error is the ( target - actual ). errors from output layer
        output_errors = targets - final_outputs
        
        #hidden layer errors
        hidden_errors = np.dot( self.weightHiddenOutput.T , output_errors )
        
        #update teh weights for the links betwen the hidden and output layers
        self.weightHiddenOutput += self.learningRate * np.dot( ( output_errors * final_outputs * ( 1.0 - final_outputs ) ) , np.transpose( hidden_outputs ) )
        
        #update the weights for the links between the input and hidden layers
        self.weightInputHidden += self.learningRate * np.dot( ( hidden_errors * hidden_outputs * ( 1.0 - hidden_outputs) ) , np.transpose( inputs ) )
    
    def query( self , input_list ) :
        
        #convert the input list to 2d array
        inputs = np.array( input_list , ndmin = 2 ).T
        
        #inputs of hidden layer
        hidden_inputs = np.dot( self.weightInputHidden , inputs )
        
        #outputs of hidden layer
        hidden_outputs = self.activationFunction( hidden_inputs )
        
        #input of output layer
        final_inputs = np.dot( self.weightHiddenOutput , hidden_outputs )
        
        #output of the hidden layer
        final_outputs = self.activationFunction( final_inputs )
        
        return final_outputs
    
    def backQuery( self , target_list ) : 
        
        FinalOutputs = np.array( target_list , ndmin = 2 ).T
        
        finalInputs = self.inverseActivationFunction( FinalOutputs )
        
        hiddenOutputs = np.dot( self.weightHiddenOutput.T , finalInputs )
        
        #scale the signal out of the hidden layer, scale back to 0.01 to 0.99
        #as the inverse of activation requires the inputs to be in the range of 0.01 to 0.99
        
        hiddenOutputs -= np.min( hiddenOutputs ) #makes the minimum values as 0
        hiddenOutputs /= np.max( hiddenOutputs ) #makes the maximum values as 1
        hiddenOutputs *= 0.98 #makes the maximum as 0.98
        hiddenOutputs += 0.01 #makes the maximum as 0.99 and minimum as 0.01
        
        hiddenInputs = self.inverseActivationFunction( hiddenOutputs )
        
        inputs = np.dot( self.weightInputHidden.T  , hiddenInputs )
        
        inputs -= np.min( inputs )
        inputs /= np.max( inputs )
        inputs *= 0.98
        inputs += 0.01
        
        return inputs
        
        
        


inputNodes = 784 #As each image has 784 values
hiddenNodes = 200
outputNodes = 10 #As there are 10 types of output labels from 0 to 9
learningRate = 0.1

n = neuralNetwork( inputNodes , hiddenNodes , outputNodes , learningRate )

#Getting training data set
trainingDataFile = open( "D:/Practice projects/ML/data files/mnist_train.csv" , 'r' )
trainingDataList = trainingDataFile.readlines()
trainingDataFile.close()

#allValues = trainingDataList[ 0 ].split( ',' )
#imageArray = np.asfarray( allValues[ 1 : ] ).reshape( ( 28 , 28 ) )
#plt.imshow( imageArray , cmap = 'Greys' , interpolation = 'None' )

#scaledInputs = ( np.asfarray( allValues[ 1 : ] ) / 255.0 * 0.99 ) + 0.01

#targets = np.zeros( outputNodes ) + 0.01
#targets[ int( allValues[ 0 ] ) ] = 0.99
#print( targets )

#train the neural network

#epochs is the number of times the training data set is used for training
epochs = 7

for e in range( epochs ) :
    
    #loop through all records in the training data set
    for record in trainingDataList : 
        
        allValues = record.split( ',' )
        
        #scale and shift the inputs
        inputs = ( np.asfarray( allValues[ 1 : ] ) / 255.0 * 0.99 ) + 0.01
        
        #allValue[ 0 ] is the target label for this record
        targets = np.zeros( outputNodes ) + 0.01
        targets[ int( allValues[ 0 ] ) ] = 0.99
        
        n.train( inputs , targets )

testDataFile = open( "D:/Practice projects/ML/data files/mnist_test.csv" , 'r' )
testDataList = testDataFile.readlines()
testDataFile.close()

#allValues = testDataList[ 0 ].split( ',' )
#print( allValues[ 0 ] )

#imageArray = np.asfarray( allValues[ 1 : ] ).reshape( ( 28 , 28 ) )
#plt.imshow( imageArray , cmap = 'Greys' , interpolation = 'None' )

#print( n.query( ( np.asfarray( allValues[ 1 : ] ) / 255.0 * 0.99 ) + 0.01 ) )

scoreCard = []

for record in testDataList : 
    
    allValues = record.split( ',' )
    
    correctLabel = int( allValues[ 0 ] )
    print( correctLabel , "Correct Label")
    
    inputs = ( np.asfarray( allValues[ 1 : ] ) / 255.0 * 0.99 ) + 0.01
    
    outputs = n.query( inputs )
    
    #gets the index of the highest value
    
    label = np.argmax( outputs )
    
    print( label , "network's answer" )
    
    if ( label == correctLabel ) :
        
        scoreCard.append( 1 )
    
    else :
        
        scoreCard.append( 0 )


scoreCardArray = np.asarray( scoreCard )
print( "Performance = " , ( scoreCardArray.sum() / scoreCardArray.size ) * 100.0 , " %" )

#for x in range( 2 ) : 
 #   print( x , "timer")
  #  time.sleep( 10 )
   # print( x )
    #backQueryLabel = np.zeros( outputNodes ) + 0.01
#    backQueryLabel[ x ] = 0.99
 #   print( backQueryLabel )
  #  inputImageArray = n.backQuery( backQueryLabel )
   # inputImageArray = ( ( inputImageArray - 0.01 ) / 0.99 ) * 255.0
    #print( "\n" , inputImageArray )
#    inputImage = np.asfarray( inputImageArray ).reshape( ( 28 , 28 ) )
 #   plt.imshow( inputImage , cmap = 'Greys' , interpolation = 'None' )
    

