from __future__ import division
import neuro
import program
#training inputs and their respective target
inputs=program.outcomes
targets=program.target

# print (inputs)
#print targets


#number of repetitions to train the network
reps=350
network=[] #makes an empty list to contain the neural net
network=neuro.setup_network(inputs)
#sets up the network to accommodate the size of your inputs
#network=neuro.readNetworkFromFile("myNetwork.net") 
#trains the network for some number of repetitions on your
#training input and targets
neuro.train(network, inputs, targets, reps)
neuro.writeNetworkToFile("myNetwork.net", network)
