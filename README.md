# Conv-Neural-Net
Code to implement a convolutional neural network. This is a very specific type of convolutional neural network, which will be used for eye-tracking in mice. The network has the following architecture: a set of filters constitutes the input layer. The filters are convolved with the input image. You can specify filter size and the number of filters (all of the same size for now). The outputs of each filter are then combined non-linearly to yield an output image, which is the size of the valid 2-D convolution between the input image and the filters. If the input image is 50x50 and the filters are 5x5, then the output image will be 46x46. 

## Goal
The goal is to track the pupil position of a mouse as it sits passively and views images or videos. The input image to the network will be an image of the mouse's eye. The output will be a slightly smaller image that depicts an approximate ellipse, which is the contour of the edge of the pupil. 
