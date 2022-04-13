import numpy as np
import scipy.special as sc #sigmoid function is called expit() in tihs library

class neuralNetwork:
    def __init__( self , inputNodes , hiddenNodes , outputNodes , learningRate ) :
        
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        
        self.learningRate = learningRate
        #the weights are preferd to be in range of -a to +a of normal distribution, where a = 1/sqrt( number of incoming links to a node )
        self.weightInputHidden = np.random.normal( 0.0 , pow( self.hiddenNodes , -0.5 ) , ( self.hiddenNodes , self.inputNodes ) )
        
        self.weightHiddenOutput = np.random.normal( 0.0 , pow( self.inputNodes , -0.5 ) , ( self.outputNodes , self.hiddenNodes ) )
        
        self.activationFunction = lambda x : sc.expit( x )
    
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
        


inputNodes = 3
hiddenNodes = 3
outputNodes = 3
learningRate = 0.5

n = neuralNetwork( inputNodes , hiddenNodes , outputNodes , learningRate )

print( n.query( [ 1.0 , 0.5 , -1.5 ] ) )