# Doodle_Xpert

https://user-images.githubusercontent.com/93630550/168848524-90bfa5ec-233b-459b-97f8-309db26e121d.mp4

A Doodle Classfier program based on Convolutional Neural Networks and a Feed forward Multilayer Perceptron. The program makes use of Google's quickdraw dataset to train and test on a sample size of 10,000 images for each doodle with an EPOCH size of 40 for better accuracy. In order to speed training - I have defined the input layer of both neural networks to be an image of size 28 X 28; However this sometimes yeilds inaccurate prediction. Due to the discrepancies with clarity.

## Currently Supported Doodles:

As the quickdraw dataset is huge, I am only using 5 doodles from it as of now but it is very easy to extend or modify them to your own needs by changing the parameters between lines 8 to 20.

```python
# Number of classes ( 5 detectable objects). Change this in case of different number of classes.
NumClasses = 5
# Dictionary of Classes and their labels. Extend or shrink depending on the number of classes.
CLASSES = {0: "Apple", 1: "Banana", 2: "Book", 3: "Cup", 4: "Ladder"}

# number of samples to take in each class. Extend or reduce depending on your system.
N = 10000

# Number of times to repeat each fruit. Change this depending on your system.
NumEPOCHS = 40

# List of files to load (in order of the dictionary). Extend or shrink depending on the number of classes.
files = ["apple", "banana", "book", "cup", "ladder"]
```

## Setting up:

Please install the pip packages mentioned in `requirements.txt` by running `pip install -r requirements.txt` in your terminal. After this you can proceed downloading the training and testing datasets in the data folder. `npy_links.txt` holds the direct download links to all currently supported doodle datasets (You can download as many or as few as you like but don't forget to modify the source code if you do so). I have also included both trained Models (CNN and MLP) with an accuracy of about 80% So you won't have the spend time training.

## Program in Action:

![CNN2](https://user-images.githubusercontent.com/93630550/168846866-acd76f5f-85d6-4f09-9020-0f59d668f97f.png) | ![CNN](https://user-images.githubusercontent.com/93630550/168847056-950dce5b-cc78-46a8-9b40-b859f40fb3c4.png)  
Using CNNs to Classify Doodles  
![MLP2](https://user-images.githubusercontent.com/93630550/168847201-3d8ec73a-53b5-45b9-9620-3d0bcc2b8a40.png) | ![MLP](https://user-images.githubusercontent.com/93630550/168847195-2c4b40c7-a6e7-47a0-bbb9-cd54317786be.png)  
Using a FNN Multi-Layer Perceptron to Classify the Doodles

## Pending Bug_Fixes:

- Further improvements in accuracy:
  - increasing the resolution
  - adding more hidden layers in the architecture
  - changing the other parameters in training such as the dataset, EPOCH size etc to make training faster
