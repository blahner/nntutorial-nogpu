import numpy as np
import matplotlib.pyplot as plt
from .transforms import InverseNormalize

def factor(num):
    """
    Factor a number to x1*x2 where x1 and x2 are as close to being
    squares of input NUM as possible. Helps for plotting various sizes of
    layers and weights.

    Parameters
    ----------
    num : int
        number that you want factored

    Returns
    -------
    factors_list : list of tuples
        Each index of FACTORS_LIST is a tuple of (x1, x2) factors of NUM

    """
    factors_list = []
    max_root = int(np.ceil(np.sqrt(num)))
    for x1 in np.arange(1, max_root+1):
        if num % x1 == 0: 
            x2 = int(num/x1)
            factors_list.append((x1, x2))
    return factors_list

def plot_sample_imgs(dataset, save_path=None):
    """
    Plots sample images from a dataset for visualization purposes and saves the plots.
    
    Parameters:
    -----------
    dataset: "mnist_numpy_dataset" class object
        dataset object of the MNIST dataset
    save_path: string, optional
        Filepath, including filename, to where you want to save the resulting plot. If not
        specified, the plot will not save.
    
    Returns
    -------
    None.
    """
    transform_inverse = InverseNormalize() #define an inverse normalize object. Since MNIST is grayscale, inverse normalizing doesn't make a big difference for visualizing. This is useful for natural images though
    fig, ax = plt.subplots(3,3)
    fig.suptitle("Sample Images from Training Set")
    for r in range(3):
        for c in range(3):
            idx = 3*r + c
            sample = dataset[idx] #with the pytorch dataset class we can just index into the dataset. It automatically calls the __getitem__ method
            img = sample["image"].reshape((28,28))
            label = sample["label"]
            
            ax[r,c].imshow(img, cmap="Greys")
            label, = np.where(label == 1)
            ax[r,c].set_title("Label: " + str(label[0]))
            ax[r,c].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.clf()

def visualize_weights(model, layer, save_path=None):
    """
    Visualize layer weights of the network in one plot. If the network is properly
    trained, we should be able to see some patterns in the weights.

    Parameters
    ----------
    param : dict
        keys are the weights and biases of each layer
    layer : int
        Use this to index which layer's weights you want to visualize
    save_path: string, optional
        Filepath, including filename, to where you want to save the resulting plot. If not
        specified, the plot will not save.

    Returns
    -------
    None.

    """
    fs = 20 #fontsize for plots
    weight_values = model.MLP[layer-1].weight.detach().cpu().numpy() # shape is (l+1 nodes, l nodes)
    rows_subplot = int(np.floor(np.sqrt(weight_values.shape[0])))
    cols_subplot = int(np.ceil(weight_values.shape[0]/rows_subplot))
    fig, ax = plt.subplots(rows_subplot, cols_subplot)
    fig.suptitle("Weight Visualization of Layer " + str(layer), fontsize = fs)
    factors_list = factor(weight_values.shape[1])
    best_factor = factors_list[-1] #factors x1 and x2 that are closest to squares
    r = best_factor[0]
    c = best_factor[1]
    print("Weight plots are " + str(r) + " pixels by " + str(c) + " pixels.")
    count = 0
    for i in np.arange(rows_subplot):
        for j in np.arange(cols_subplot):
            if count < weight_values.shape[0]:
                val = weight_values[count, :]
                ax[i,j].imshow(val.reshape(r,c))
                count += 1
            ax[i,j].axis(False)
        
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.clf()
    
def visualize_activations(activation_values, save_path=None):
    """
    Visualize the activations of an image as it gets passed through the network.
    Note how the activations change depending on whether the network is trained
    or not. 

    Parameters
    ----------
    activation_values : Dictionary
        Dictionary of activation values for each image at each layer in the model.
        Be conscious of the 'activation_values' that you pass in here. If using
        the recommended method 'model.get_activations()', the activations are 
        from the most recent pass of images through the model. It could be training,
        validation, or test.
    save_path: string, optional
        Filepath, including filename, to where you want to save the resulting plot. If not
        specified, the plot will not save.

    Returns
    -------
    None.

    """
    for key in activation_values.keys():
        act = activation_values[key]
        shape = act.shape[0]
        if shape == 10: #output layer
            best_factor = (1,10)
        else:
            factors_list = factor(act.shape[0])
            best_factor = factors_list[-1] #factors x1 and x2 that are closest to squares
        r = best_factor[0]
        c = best_factor[1]
        act_reshape = act.reshape((r,c))
        plt.imshow(act_reshape)
        if shape == 10:
            for j in np.arange(10):
                plt.text(j, 0, round(act_reshape[0,j].item(),2), ha="center", va="center", color="k")
            plt.xticks(ticks = np.arange(10), labels = np.arange(10))
            plt.yticks(ticks=[])
            plt.title("MNIST Digit Confidence")
        else:
            plt.title("Reshaped Activation of " + key)        
        if save_path:
            plt.savefig(save_path + "_" + key + ".png")
        plt.show()
        plt.clf()
