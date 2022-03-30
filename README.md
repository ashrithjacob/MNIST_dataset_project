### MNIST_dataset_project

Digit recognition projects on the MNIST dataset
The MNIST dataset contains numbers from 0-9 in freehand writing.
The total dataset contains 70000 images with each image being a 28x28 pixel image
We shall explore some simple neural nets through this dataset and implement our custom made neural nets for the same:

In this project we use 2 hidden layers and an output softmax layer containing 10 nodes.
We use an input layer containing a single channel from the greyscale image(0-255)
here's a heuristic of the Network used:
![Digit_rec](https://github.com/ashrithjacob/MNIST_dataset_project/blob/main/Digit_rec.png)

For the loss function, we use a negative likelihood loss function and for the gradient descent a SGD is used.



