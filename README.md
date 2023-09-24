# Overview
This self-paced tutorial provides a comprehensive, one-stop resource for understanding the fundamentals of neural networks without the need for GPUs or internet access.  This tutorial series is split into five parts:
1. an overview of neural networks
2. the math behind forward and back propagation
3. MNIST classification with a MLP in NumPy
4. MNIST classification with a MLP in PyTorch
5. MNIST classification with a CNN in PyTorch


Along the way, I provide offline PDF resources with more detailed and/or supplemental information.
# Who is this tutorial for?
This tutorial is for beginner programmers who want to understand neural networks. You should have a good working knowledge of python. An introductory python course (like [MIT's free 6.0001 course](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/syllabus/)) or similar would be perfect. You do not need any internet access after completing the following "Getting Started" steps, and your computer does not need any GPUs.

# Getting started
After completing the following steps, you will be able to get working with no need for anymore internet access or GPUs.

## Step 01: Install Python and packages
I will mention two ways to download Python. Python can installed either
1. via Python's [website](https://www.python.org/downloads/) or
2. via Anaconda's [website](https://www.anaconda.com/products/individual).


The difference between these two installations is that downloading Python from the Python website (option 1) only downloads Python and nothing else. Anaconda (option 2) downloads Python plus many other (useful) software, including a nice IDE (Spyder) to make it easier to write and debug code. So which Python installation method should you choose? Anaconda is a comprehensive data science platform that includes a lot of useful software for all things data science, so if you think you might want to continue learning about data science and/or machine learning, download Anaconda. Downloading Anaconda is my recommendation. Install Python from Python's website if computer storage is an issue (Anaconda will take up about 5GB on your computer because of all the extra software). DO NOT download both Anaconda and Python, as this will result in having two versions of Python on your computer and will be confusing. If you run into issues, the course materials for the [MIT 6.0001 course](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/syllabus/) 
(specifically the setup instructions in [Problem Set 0](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/assignments/)) provide instructions for downloading Python via Anaconda. This course material is also included in the external resources you will download in Step 04 below.


Once Python is installed (either via Anaconda or Python's website), you will need to install the numpy, matplotlib, copy, os, and torch packages to Python. To do so, in your computer's command line type "pip install <package>" (without the quotes) for all packages, where <package> is either numpy, matplotlib, copy, os, or torch. To verify the packages installed correctly, open Python and in your console type "import <package>". If you see no errors, that means the packages installed correctly.

## Step 02: Download this GitHub project code and tutorial walkthrough
Code download: Click on the green dropdown button that says "code" in the upper right corner and click "Download as .zip". This will download the tutorial code.

Tutorial walkthrough: Click [here](TODO) to download the PDF walkthrough of the above code.

## Step 03: Download the MNIST dataset
This tutorial uses the MNIST dataset, a collection of handwritten digits from 0-9. Download the MNIST .csv files [here](https://drive.google.com/drive/folders/1prkKWdSNq_SK_q5Duj-hyh4Y0nvXrlcH?usp=drive_link). You can learn more about MNIST [here](http://yann.lecun.com/exdb/mnist/).

## Step 04: Download the external PDF resources
Download all external resources [here](https://drive.google.com/drive/folders/1OQEgalDeaHa5KrD6NLxP5t8pGGQu--OI?usp=sharing)

# Project Organization
Move all the files and folders you downloaded into a root project directory of your choice. Organize the directories under your root project directory into this file structure:


/path/to/your/project/

├── CNN_pytorch
│   ├── model
│   └── utils
├── MLP_numpy
│   ├── model
│   └── utils
├── MLP_pytorch
│   ├── model
│   └── utils
├── data
├── data_preparation
└── resources
    ├── Documentation
    ├── intuition
    ├── math
    ├── numpy_code
    ├── PythonRefresher_MIT6.0001
    └── pytorch_code


# What's next?
After this tutorial, you will have an excellent understanding of the fundamentals of neural networks and are ready to try our more advanecd models (e.g., ResNet, transformers, mobilenets), larger datasets (e.g., CIFAR10, ImageNet, COCO), and other modalities (e.g., text, audio). [PyTorch](https://pytorch.org/tutorials/) has an awesome tutorials page and is a great place to start. Beyond this tutorial, however, GPUs and internet access will pretty much become necessary. For free GPUs, you can use [Google CoLab](https://colab.research.google.com/). 
