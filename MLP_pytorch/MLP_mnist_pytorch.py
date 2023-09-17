#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: benjamin lahner
"""
import copy
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

#below are local imports. We could have defined everything in one script, but the code would have been lengthy and hard to read
from model.architectures import MLP_pytorch  # imports the MLP class we define in Step #3
from utils.dataset import mnist_pytorch_dataset  # import our custom dataset class
from utils.helper import plot_sample_imgs, visualize_activations, visualize_weights
from utils.transforms import Normalize, ToTensor1D

def train(model, dataloaders, criterion, optimizer, device, epochs=3):
    """
    Core training function for your network. This function feeds the images and labels
    into the neural network, computes the loss, and updates the neural network weights.
    The test set should not be passed through here. This function utilizes some pytorch
    utilities to make training easier.

    Parameters
    ----------
    model : 'MLP_pytorch' class object
        the defintion of the multi-layer perceptron model from the 'MLP_pytorch'
        class defined in architectures.py
    dataloaders : dictionary
        dictionary with "train" and "val" keys and values of the pytorch "Dataloader" class
    criterion : "MSELoss" object class
        An instance of our custom "MSELoss" class that defines the network's cost function
    optimizer : "optim" object class
        Instance of the optim object class that defines how the network weights get updated through
        backpropogation.
    device : string
        A string returned by "torch.device()" that determines whether the computations will take
        place on the computer's cpu or gpu. GPU is always faster. For this tutorial, either is fine.
        For larger models and/or datasets, you will want a GPU.
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
    best_model_wts = copy.deepcopy(model.state_dict()) #initially this is the model's randomly initialized weights.
    best_acc = 0
        
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs-1))
        print('-'*10)
        
        #Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() #set model to training mode i.e. keep track of gradients
            elif phase == 'val':
                model.eval() #set model to training mode
                
            running_loss = 0.0
            running_corrects = 0
                
            dataloader = dataloaders[phase]
            #iterate over data
            for sample in dataloader:
                inputs = sample["image"].to(device)
                labels = sample["label"].to(device)
                    
                optimizer.zero_grad() #zero out the parameter gradients
                
                #forward pass, only track history in train phase
                with torch.set_grad_enabled(phase=='train'):
                    #get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    #backprop and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == torch.argmax(labels.data,axis=1)).item()
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)
            print("{} Loss: {:4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)    
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_loss_history.append(epoch_loss)    
                train_acc_history.append(epoch_acc)       
                    
    #load best model weights
    model.load_state_dict(best_model_wts)
    return val_acc_history, train_acc_history, val_loss_history, train_loss_history 

def inference(model, dataloader, criterion, device):
    """
    Run inference using this function. This function does not train
    the network (so no need to specify any learning rate or epochs or validation sets), but merely computes the output
    No need to shuffle the test set for evaluation.
    Parameters
    ----------
    model : 'MLP_pytorch' class object
        the defintion of the multi-layer perceptron model from the 'MLP_pytorch'
        class defined in the architectures.py file
    dataloader : instance of pytorch Dataloader class
        Instance of pytorch Dataloader class that defines how images and labels
         are fed into the network
    criterion : "MSELoss" object class
        An instance of our custom "MSELoss" class that defines the network's cost function 
    device : string
        A string returned by "torch.device()" that determines whether the computations will take
        place on the computer's cpu or gpu. GPU is always faster. For this tutorial, either is fine.
        For larger models and/or datasets, you will want a GPU.
    Returns
    -------
    inference_loss : float
        Loss of the model (usually from the test set) after inference
    inference_acc : float
        Accuracy of the model (usually from the test set) after inference. Output 
        is a decimal 0-1. Multiply by 100 to get percent
    """
    running_loss = 0
    running_correct = 0
        
    print("Running Model Inference")
    print('-'*10)
        
    model.eval()
    #iterate over data
    for sample in dataloader:
        inputs = sample["image"].to(device)
        labels = sample["label"].to(device)

        with torch.set_grad_enabled(False): #we will never use this 'inference' to train the model, so we can set this to false
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
        running_loss += loss.item() #scale loss by how number of images in the batch
        running_correct += torch.sum(preds == torch.argmax(labels,axis=1)).item()
      
    inference_loss = running_loss/len(dataloader.dataset)
    inference_acc = running_correct/len(dataloader.dataset)
    
    print("Inference Loss: {:4f} Acc: {:.4f}".format(inference_loss, inference_acc))
    return inference_loss, inference_acc

if __name__ == '__main__':

    #########################
    #STEP 0: Check to see if we have a GPU to use. It's ok if not! 
    use_cuda = torch.cuda.is_available()
    print("Using {}".format('GPU' if use_cuda else 'CPU'))
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only")

    #Define our paths
    root = os.path.join("/home","blahner","projects","nntutorial")
    data_root = os.path.join(root, "data") #path to MNIST data
    save_root = os.path.join(root, "MLP_pytorch", "output") #to save figures and other output

    #make the folder to save output if it doesn't exist already
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    #########################
    #STEP 1: Load data into a dataset and visualize samples for a sanity check
    dataset_train_viz = mnist_pytorch_dataset(data_root, phase='train', transforms=None) # a dataset with no transforms applied just so we can visualize the raw images
    plot_sample_imgs(dataset_train_viz, save_path=os.path.join(save_root, "training_samples.png"))
    
    #Let's use some pytorch-defined transforms
    transform = transforms.Compose([ToTensor1D(), Normalize()]) #we want to make these input images tensors so we can use pytorch to calculate gradients
    dataset_train = mnist_pytorch_dataset(data_root, phase='train', transforms=transform)
    dataset_val = mnist_pytorch_dataset(data_root, phase='val', transforms=transform)
    dataset_test = mnist_pytorch_dataset(data_root, phase='test', transforms=transform)

    print("Number of Training Images:", len(dataset_train))
    print("Number of Validation Images:", len(dataset_val))
    print("Number of Testing Images:", len(dataset_test))

    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=1)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    dataloaders = {'train': dataloader_train, 'val': dataloader_val}

    #########################
    #STEP 2: Define the MLP model, criterion, and optimizer
    layer_sz = [784, 144, 10] #Each element of layer_sz represents the size of that layer.
    model = MLP_pytorch(layer_sz).to(device) #initialize your MLP model and put it on device
    print(model)
    
    criterion = nn.MSELoss() #use pytorch's mean squared error loss criterion
    
    #You are essentially giving your model's parameters as inputs
    #to this "optim.SGD" class to tell it how each one should be updated. SGD, or Stochastic Gradient Descent, is 
    #mathematically the most straightforward, commonly used, and the one explained in depth in this tutorial. Other
    #optimizers, like Adam, are similar (still gradient based), but add a bit more functionality useful for large networks. 
    optimizer = optim.SGD(model.parameters(), lr=0.5) 

    #########################
    #STEP 3: let's examine the model's accuracy, weights, and image activations before any training happens
    print("Model Performance Before Training")
    inference_loss, inference_acc = inference(model, dataloader_test, criterion, device) #pass the testing set through the network
    visualize_weights(model, 1, save_path=os.path.join(save_root, "modelweights_training-before.png")) #Do you see any structure in these weights?
    
    # get activations from a sample
    feature_extractor_all_layers = list(model.children())[0] #gets all the layers of the network (children)
    sample = dataset_test[5] #grab any arbitrary sample from the dataset to get the activations from
    x = sample["image"].to(device) #put the image on the proper device so we can pass it through layers of the network again
    activations = {} #stores the activations
    activations["Layer0"] = x.detach().cpu().numpy() #start with the pure image
    for count, extract_layer in enumerate(feature_extractor_all_layers):
        if isinstance(extract_layer, nn.Sigmoid): #we want to see the activations after the sigmoid nonlinearity
            subnet = feature_extractor_all_layers[0:count+1] #define a sub network that contains all layers from the beginning up to the current one
            out = subnet(x) #pass the sample image through the subnet
            activations["Layer" + str(count)] = out.detach().cpu().numpy()
   
    visualize_activations(activations, save_path=os.path.join(save_root, "modelactivations_training-before") )

    #########################
    #STEP 4: Train the model! Most of the code's runtime will be spent here.
    val_acc, train_acc, val_loss, train_loss = train(model, dataloaders, criterion, optimizer, device, epochs=3)
    
    #Plot the training and validation accuracy and loss. Essential for debugging your network
    #plot training and validation cost resutls
    plt.plot(train_loss), plt.plot(val_loss)
    plt.ylabel("MSE Loss"), plt.xlabel("Epoch"), plt.ylim(bottom=0)
    plt.title("MLP Training Loss Curves")
    plt.legend(["Training","Validation"])
    plt.savefig(os.path.join(save_root, "training_loss_curves.png"))
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
    inference_loss, inference_acc = inference(model, dataloader_test, criterion, device) #pass the testing set through the network
    visualize_weights(model, 1, save_path=os.path.join(save_root, "modelweights_training-after.png")) #Do you see any structure in these weights?

    # get activations from a sample
    feature_extractor_all_layers = list(model.children())[0] #gets all the layers of the network (children)
    sample = dataset_test[5] #grab any arbitrary sample from the dataset to get the activations from
    x = sample["image"].to(device) #put the image on the proper device so we can pass it through layers of the network again
    activations = {} #stores the activations
    activations["Layer0"] = x.detach().cpu().numpy() #start with the pure image
    for count, extract_layer in enumerate(feature_extractor_all_layers):
        if isinstance(extract_layer, nn.Sigmoid): #we want to see the activations after the sigmoid nonlinearity
            subnet = feature_extractor_all_layers[0:count+1] #define a sub network that contains all layers from the beginning up to the current one
            out = subnet(x) #pass the sample image through the subnet
            activations["Layer" + str(count)] = out.detach().cpu().numpy()
   
    visualize_activations(activations, save_path=os.path.join(save_root, "modelactivations_training-after"))