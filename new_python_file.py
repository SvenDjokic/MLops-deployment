# %% [markdown]
# # Lab : Image Classification using Convolutional Neural Networks
# 
# At the end of this laboratory, you would get familiarized with
# 
# *   Creating deep networks using Keras
# *   Steps necessary in training a neural network
# *   Prediction and performance analysis using neural networks
# 
# ---

# %% [markdown]
# # **In case you use a colaboratory environment**
# By default, Colab notebooks run on CPU.
# You can switch your notebook to run with GPU.
# 
# In order to obtain access to the GPU, you need to choose the tab Runtime and then select “Change runtime type” as shown in the following figure:
# 
# ![Changing runtime](https://miro.medium.com/max/747/1*euE7nGZ0uJQcgvkpgvkoQg.png)
# 
# When a pop-up window appears select GPU. Ensure “Hardware accelerator” is set to GPU.

# %% [markdown]
# # **Working with a new dataset: CIFAR-10**
# 
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. More information about CIFAR-10 can be found [here](https://www.cs.toronto.edu/~kriz/cifar.html).
# 
# In Keras, the CIFAR-10 dataset is also preloaded in the form of four Numpy arrays. x_train and y_train contain the training set, while x_test and y_test contain the test data. The images are encoded as Numpy arrays and their corresponding labels ranging from 0 to 9.
# 
# Your task is to:
# 
# *   Visualize the images in CIFAR-10 dataset. Create a 10 x 10 plot showing 10 random samples from each class.
# *   Convert the labels to one-hot encoded form.
# *   Normalize the images.
# 
# 
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# %%
# Your code here :

# %% [markdown]
# ## Define the following model (same as the one in tutorial)
# 
# For the convolutional front-end, start with a single convolutional layer with a small filter size (3,3) and a modest number of filters (32) followed by a max pooling layer. 
# 
# Use the input as (32,32,3). 
# 
# The filter maps can then be flattened to provide features to the classifier. 
# 
# Use a dense layer with 100 units before the classification layer (which is also a dense layer with softmax activation).

# %%
from keras.backend import clear_session
clear_session()

# %%
# Your code here :

# %% [markdown]
# *   Compile the model using categorical_crossentropy loss, SGD optimizer and use 'accuracy' as the metric.
# *   Use the above defined model to train CIFAR-10 and train the model for 50 epochs with a batch size of 512.

# %%
# Your code here :

# %% [markdown]
# *   Plot the cross entropy loss curve and the accuracy curve

# %%
# Your code here :

# %% [markdown]
# ## Defining Deeper Architectures: VGG Models
# 
# *   Define a deeper model architecture for CIFAR-10 dataset and train the new model for 50 epochs with a batch size of 512. We will use VGG model as the architecture.
# 
# Stack two convolutional layers with 32 filters, each of 3 x 3. 
# 
# Use a max pooling layer and next flatten the output of the previous layer and add a dense layer with 128 units before the classification layer. 
# 
# For all the layers, use ReLU activation function. 
# 
# Use same padding for the layers to ensure that the height and width of each layer output matches the input
# 

# %%
from keras.backend import clear_session
clear_session()

# %%
# Your code here :

# %% [markdown]
# *   Compile the model using categorical_crossentropy loss, SGD optimizer and use 'accuracy' as the metric.
# *   Use the above defined model to train CIFAR-10 and train the model for 50 epochs with a batch size of 512.

# %%
# Your code here :

# %% [markdown]
# *   Compare the performance of both the models by plotting the loss and accuracy curves of both the training steps. Does the deeper model perform better? Comment on the observation.
#  

# %%
# Your code here :

# %% [markdown]
# **Comment on the observation**
# 
# *(Double-click or enter to edit)*
# 
# ...

# %% [markdown]
# *   Use predict function to predict the output for the test split
# *   Plot the confusion matrix for the new model and comment on the class confusions.
# 

# %%
# Your code here :

# %% [markdown]
# **Comment here :**
# 
# *(Double-click or enter to edit)*
# 
# ...

# %% [markdown]
# *    Print the test accuracy for the trained model.

# %%
# Your code here :

# %% [markdown]
# ## Define the complete VGG architecture.
# 
# Stack two convolutional layers with 64 filters, each of 3 x 3 followed by max pooling layer. 
# 
# Stack two more convolutional layers with 128 filters, each of 3 x 3, followed by max pooling, followed by two more convolutional layers with 256 filters, each of 3 x 3, followed by max pooling. 
# 
# Flatten the output of the previous layer and add a dense layer with 128 units before the classification layer. 
# 
# For all the layers, use ReLU activation function. 
# 
# Use same padding for the layers to ensure that the height and width of each layer output matches the input
# 
# *   Change the size of input to 64 x 64.

# %%
from keras.backend import clear_session
clear_session()

# %%
# Your code here :

# %% [markdown]
# *   Compile the model using categorical_crossentropy loss, SGD optimizer and use 'accuracy' as the metric.
# *   Use the above defined model to train CIFAR-10 and train the model for 10 epochs with a batch size of 512.
# *   Predict the output for the test split and plot the confusion matrix for the new model and comment on the class confusions.

# %%
# Your code here :

# %% [markdown]
# # Understanding deep networks
# 
# *   What is the use of activation functions in network? Why is it needed?
# *   We have used softmax activation function in the exercise. There are other activation functions available too. What is the difference between sigmoid activation and softmax activation?
# *   What is the difference between categorical crossentropy and binary crossentropy loss?

# %% [markdown]
# **Write the answers below :**
# 
# 1 - Use of activation functions:
# 
# 
# 
# _
# 
# 2 - Key Differences between sigmoid and softmax:
# 
# 
# 
# _
# 
# 3 - Key Differences between categorical crossentropy and binary crossentropy loss:
# 
# 
# _
# 


