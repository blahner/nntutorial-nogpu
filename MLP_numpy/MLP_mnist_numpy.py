#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benjamin lahner
"""
import os

import matplotlib.pyplot as plt
import numpy as np

#below are local imports. We could have defined everything in one script, but the code would have been lengthy and hard to read
from model.architectures import MLP_numpy  # imports the MLP class we define in Step #3
from utils.dataloader import mnist_numpy_dataloader  # import our custom dataloader class
from utils.dataset import mnist_numpy_dataset  # import our custom dataset class
from utils.helper import (plot_sample_imgs, visualize_activations, visualize_weights)
from utils.loss import MSELoss  # import custom loss function
from utils.transforms import Normalize  # import our custom transforms classes

def train(model, dataloaders, criterion, learning_rate=0.5, epochs=3):
    """
    Put it all together and train your network! This function manages the training and 
    validation epochs. The test set should not be passed through here.

    Parameters
    ----------
    model : 'MLP_numpy' class object
        the defintion of the multi-layer perceptron model from the 'MLP_numpy'
        class defined in architectures.py
    dataloaders : dictionary
        dictionary with "train" and "val" keys and values of the "dataloader" class
    criterion : "MSELoss" object class
        An instance of our custom "MSELoss" class that defines the network's cost function
    learning_rate : float, optional
        learning rate of training. Defines how much to update the weights each backprop step.
        Defaults to 0.5
    epochs : int, iptional
        number of epochs to train the model for. Each epoch does a complete pass through all
        training images. Defaults to 3 epochs

    Returns
    -------
    train_loss_history : list of float
        total training cost at each epoch
    train_acc_history : list of float
        total training accuracy at each epoch
    val_loss_history : list of float
        total validation cost at each epoch
    val_acc_history : list of float
        total validation accuracy at each epoch

    """
    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []
        
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs-1))
        print('-'*10)
        
        #Each epoch has a training and validation phase
        for phase in ['train', 'val']:   
            running_loss = 0.0
            running_corrects = 0
                
            dataloader = dataloaders[phase]
            #iterate over data
            for inputs, labels in dataloader:
                #forward pass, only track history in train phase
                outputs = model.forward_all_layers(inputs)
                
                loss = criterion(outputs, labels)
                    
                #update the loss and accuracies
                running_loss += loss
                running_corrects += np.sum(np.argmax(outputs, axis=0) == np.argmax(labels,axis=1))
                
                if phase == 'train':
                    #backward pass only when training
                    dc_da = criterion.derivative(outputs, labels) # get the derivative of the cost w.r.t. the activation
                    model.backward_all_layers(dc_da, learning_rate) #update weights and biases on each batch
                    #model.backward_all_layers(outputs, labels.T, learning_rate) #update weights and biases on each batch
            
            #divide the running loss and accuracy by the total number of images to approximate the loss for the epoch
            epoch_loss = running_loss/len(dataloader.dataset)
            epoch_acc = running_corrects/len(dataloader.dataset)
            print("{} Loss: {:4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)    
                train_acc_history.append(epoch_acc) 
            if phase == 'val':
                val_loss_history.append(epoch_loss)    
                val_acc_history.append(epoch_acc)
    
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history 

def inference(model, dataloader, criterion):
    """
    Run inference using this function. This function does not train
    the network (so no need to specify any learning rate or epochs or validation sets), but merely computes the output
    No need to shuffle the test set for evaluation.
    Parameters
    ----------
    model : 'MLP_numpy' class object
        the defintion of the multi-layer perceptron model from the 'MLP_numpy'
        class defined in the architectures.py file
    dataloader : instance of "mnist_numpy_dataloader" class
        Iterable instance of "mnist_numpy_dataloader" class that defines how images and labels
         are fed into the network
    criterion : "MSELoss" object class
        An instance of our custom "MSELoss" class that defines the network's cost function 

    Returns
    -------
    inference_loss : float
        Loss of the model (usually from the test set) after inference
    inference_acc : float
        Accuracy of the model (usually from the test set) after inference. Output 
        is a decimal 0-1. Multiply by 100 to get percent
    activations : dict of numpy arrays
        Dictionary where keys are each layer in the model and values are arrays
        of size (n_classes x n_images) where n_images is the number of images in 'test_imgs'.
        Intended to collect all model activations from all batches.

    """

    running_loss = 0.0
    running_corrects = 0

    print("Running Model Inference")
    print('-'*10)

    num_imgs = len(dataloader.dataset)
    activations = {"layer" + str(c): np.zeros((lay, num_imgs)) for c,lay in enumerate(model.get_layer_sz())}

    count = int(0)
    #iterate over data
    for inputs, labels in dataloader:
        #forward pass, only track history in train phase
        outputs = model.forward_all_layers(inputs)
        loss = criterion(outputs, labels)
            
        act_tmp = model.get_activations() #activations for that batch
        #this loop accumulates the activations over all epochs before they get overwritten
        for key, val in act_tmp.items():
            activations[key][:,count:(count + inputs.shape[0])] = val
        count += inputs.shape[0]

        #update the loss and accuracies
        running_loss += loss
        running_corrects += np.sum(np.argmax(outputs, axis=0) == np.argmax(labels,axis=1))
    inference_loss = running_loss/len(dataloader.dataset)
    inference_acc = running_corrects/len(dataloader.dataset)
    print("Inference Loss: {:4f} Acc: {:.4f}".format(inference_loss, inference_acc))

    return inference_loss, inference_acc, activations

if __name__ == '__main__':
    #Define our paths
    root = os.path.join("/home","blahner","projects","nntutorial")
    data_root = os.path.join(root, "data") #path to MNIST data
    save_root = os.path.join(root, "MLP_numpy", "output") #to save figures and other output

    #make the folder to save output if it doesn't exist already
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    #########################
    #STEP 1: Load data into a dataset and visualize samples for a sanity check
    dataset_train_viz = mnist_numpy_dataset(data_root, phase='train', transforms=None) # a dataset with no transforms applied just so we can visualize the raw images
    plot_sample_imgs(dataset_train_viz, save_path=os.path.join(save_root, "training_samples.png"))

    transform = [Normalize()] #normalizing the images helps with training
    dataset_train = mnist_numpy_dataset(data_root, phase='train', transforms=transform)
    dataset_val = mnist_numpy_dataset(data_root, phase='val', transforms=transform)
    dataset_test = mnist_numpy_dataset(data_root, phase='test', transforms=transform)

    print("Number of Training Images:", len(dataset_train))
    print("Number of Validation Images:", len(dataset_val))
    print("Number of Testing Images:", len(dataset_test))

    dataloader_train = mnist_numpy_dataloader(dataset_train, batch_sz=16, shuffle=True)
    dataloader_val = mnist_numpy_dataloader(dataset_val, batch_sz=16, shuffle=False)
    dataloader_test = mnist_numpy_dataloader(dataset_test, batch_sz=1, shuffle=False)

    dataloaders = {'train': dataloader_train, 'val': dataloader_val}

    #########################
    #STEP 2: Define the MLP model and criterion (cost function).
    # Navigate to model --> architectures.py file to see the MLP class definition
    # Note that in this numpy implementation, our "optimizer" (i.e. using 
    # Stochastic Gradient Descent to update weights) is baked into the network. This is
    # typically not the case. You will see in the later Pytorch implementations that we 
    # define an optimizer seperately from the model.
    
    layer_sz = [784, 144, 10] #Each element of layer_sz represents the size of that layer.
    model = MLP_numpy(layer_sz) #initialize your MLP model
    print(model)

    criterion = MSELoss() #how will the network be punished/rewarded? This is the MSE Loss function that we defined

#########################
    #STEP 3: let's examine the model's accuracy, weights, and image activations before any training happens
    #We should have an accuracy of about chance (~0.1 or ~10%) for our 10 MNIST digit classes.
    #The model weights and activations should have no obvious structure
    print("Model Performance Before Training")
    inference_loss, inference_acc, activations = inference(model, dataloader_test, criterion) #pass the testing set through the network
    visualize_weights(model.get_parameters(), 1, save_path= os.path.join(save_root,"modelweights_training-before.png")) #Do you see any structure in these weights?
    visualize_activations(activations, layers=list(range(len(layer_sz))), 
                    image_idx=0, save_path=os.path.join(save_root,"modelactivations_training-before"))

    #########################
    #STEP 4: Train the model! Most of the code's runtime will be spent here.
    train_loss, train_acc, val_loss, val_acc = train(model, dataloaders, criterion, learning_rate=0.5, epochs=10)

    #Plot the training and validation accuracy and loss. Essential for debugging your network
    #plot training and validation cost resutls
    plt.plot(train_loss), plt.plot(val_loss)
    plt.ylabel("MSE Loss"), plt.xlabel("Epoch"), plt.ylim(bottom=0)
    plt.title("MLP Training Loss Curves")
    plt.legend(["Training","Validation"])
    plt.savefig(os.path.join(save_root,"training_loss_curves.png"))
    plt.show()
    plt.clf()

    #plot training and validation accuracy resutls
    plt.plot(range(0,len(train_acc)), train_acc), plt.plot(range(0,len(val_acc)), val_acc)
    plt.title("MLP Training Accuracy Curves")
    plt.ylabel("Accuracy"), plt.xlabel("Epoch"), plt.ylim(top=1)
    plt.legend(["Training","Validation"])
    plt.savefig(os.path.join(save_root, "training_acc_curves.png"))
    plt.show()
    plt.clf()

    #########################
    #STEP 5: let's examine the model's accuracy, weights, and image activations after training. See any differences to the before training?
    
    print("Model Performance After Training")
    inference_loss, inference_acc, activations = inference(model, dataloader_test, criterion) #pass the testing set through the network
    visualize_weights(model.get_parameters(), 1, save_path=os.path.join(save_root, "modelweights_training-after")) #Do you see any structure in these weights?
    visualize_activations(activations, layers=list(range(len(layer_sz))), 
                    image_idx=0, save_path=os.path.join(save_root, "modelactivations_training-after"))

    #########################
    #STEP 6 (extra): #Where did our model have some difficulty? We can look into the activations of
    #the final layer and, if we interpret the activation values as how confident the model
    #is in its classification, we can see which images the model classified with low confidence.
    #Run these lines again to see multiple examples of the images the model had 
    #difficulty classifying

    difficult_imgs_idx = np.where(np.max(activations['layer' + str(len(layer_sz)-1)],0) < 0.3) #the 0.3 value is somewhat arbitrary
    difficult_img_idx = np.random.choice(difficult_imgs_idx[0]) #choose a random 'difficult image' to examine

    #plot the difficult image with its label
    plt.imshow(activations['layer0'][:,difficult_img_idx].reshape((28,28)), cmap="Greys")
    label, = np.where(dataset_test.getitem(difficult_img_idx)["label"] == 1)
    plt.title("True Label: " + str(label.item()),fontsize=30)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_root, "difficultimg.png"))
    plt.show()
    plt.clf()

    #Visualize the confidence of the model's classification. Does the model's
    #lack of confidence make sense based on the image? 
    visualize_activations(activations, layers=[len(layer_sz)-1], 
                        image_idx=difficult_img_idx, save_path=os.path.join(save_root, "modelactivations_difficultimg"))

    #########################
    """
    #STEP 7: Next steps! Once you see that your network can learn, experiment with different 
    hyperparameters to see how it affects learning. Only change one hyperparameter 
    at a time so you are sure that any change in network performance is due to that 
    one hyperparameter! Try the following and note the network performance 
    (testing accuracy and train/val accuracy/loss curves) and training time:
    -Vary the batch size
    -Vary the number of epochs
    -Vary the learning rate
    -In a network with one hidden layer, vary the hidden layer depth
    -Vary the number of layers in a network
    -Initialize the weights to all zeros instead of random numbers
    -Change the train/val split (e.g. 20% train, 80% validation)
    -Train on shuffled labels so the labels don't correspond to the correct image

    As another exercise or to help get started, train a network on the OR gate.
    The OR gate is a very simple nonlinear function. Remember to change the "assert"
    statement in the MLP definition if you're not using MNIST'

    OR Gate:
        Parameters:
            layer_sz: list of int
                define the size and number of network layers
            x: numpy array
                input to model (analogous to the MNIST images)
            y: numpy array
                input labels (analogous to the MNIST labels).
                
        
    layer_sz = [2, 3, 1]
    x = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
    y = np.array([[0],
                [1],
                [1],
                [1]])
    """