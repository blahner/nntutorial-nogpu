# Overview
This self-paced tutorial provides a comprehensive, one-stop resource for understanding the fundamentals of neural networks without the need for GPUs or internet access.  This tutorial series is split into five parts:
1. an overview of neural networks
2. the math behind forward and back propagation
3. MNIST classification with a MLP in NumPy
4. MNIST classification with a MLP in PyTorch
5. MNIST classification with a CNN in PyTorch


Along the way, I provide offline PDF resources with more detailed and/or supplemental information.

# Getting started
After completing the following steps, you will be able to get working with no need for anymore internet access or GPUs.

## Step 01: Install Python and packages
I will mention two ways to download Python. Python can installed either 1. via Python's [website](\href{https://www.python.org/downloads/) or 2. via Anaconda's [website](\href{https://www.anaconda.com/products/individual). The difference between these two installations is that downloading Python from the Python website only downloads Python and nothing else. Anaconda downloads Python plus many other (useful) software, including a nice IDE (Spyder) to make it easier to write and debug code. So which Python installation method should you choose? Anaconda is a comprehensive data science platform that includes a lot of useful software for all things data science, so if you think you might want to continue learning about data science and/or machine learning, download Anaconda. Downloading Anaconda is my recommendation. Install Python from Python's website if computer storage is an issue (Anaconda will take up about 5GB on your computer because of all the extra software). DO NOT download both Anaconda and Python, as this will result in having two versions of Python on your computer and will be confusing. If you run into issues, the course materials for the [MIT 6.0001 course](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/syllabus/) 
(specifically the setup instructions in [Problem Set 0](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/pages/assignments/)) provide instructions for downloading Python via Anaconda.


Once Python is installed (either via Anaconda or Python's website), you will need to install the NumPy, matplotlib, copy, os, and torch packages to Python. To do so, in your computer's command line type "pip install $<$package$>$" for all packages, where $<$package$>$ is either NumPy, matplotlib, copy, os, or torch. To verify the packages installed correctly, open Python and in your console type "import $<$package$>$". If you see no errors, that means the packages installed correctly.
