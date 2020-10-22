# Lab2 Image Mining

## - Deep Learning with PyTorch: CIFAR10 object classification

By Jiangnan HUANG & Thomas SU - 21/10/2020

Instructor: Gianni Franchi, Antoine Manzanera

Here is [*our notebook*](https://github.com/JiangnanH/ImageMining/blob/master/Lab2/Copy_of_PyTorch_cifar10_tutorial_ROB313_2020.ipynb).

### 1. Explain the neural networks and hyper parameters configurations we tested and the resulting performances and trade-offs found.


**Neural networks:**

We've tested CNN based model with different structures.

- *kernel size:*   
  we've tested kernel size as 3x3, 5x5, 7x7 for each convolution layer, and 3x3 remains the best kernel size.
  
- *number of feature maps:*  
  we've tested several kinds of feature maps, which contains:  
  32 -> 48 -> 64 -> 128  
  32 -> 64 -> 128 -> 256  
  32 -> 64 -> 128 -> 256 -> 512  
  etc.  
  
  The best performance is given by:  
  32 -> 64 -> 128 -> 256
  
- *number of convolutional layers:*  
  we've tested 3, 6, 9, 12 convolution layers, and with 6 convolution layers the model could obtain the best performance.

- *size of fully connected layers:*  
  we've tested:  
  256 -> 10  
  64 -> 10  
  256 -> 64 -> 10  
  etc.
  
  The best performance is given by:  
  256 -> 64 -> 10

- *use dropout or not(also number of dropout layers):*  
  add dropout layer could reduce the risk of overfitting and make the model more robust, but too much dropout may cause the model converge much slower. Finally we added 1 dropout layer in our model between 2 fully connected layers.

**Hyper parameters:**

- *batch size:*  
  we test batch_size = 32, 64, 128, 256, 512. With small batch size (batch_size = 32), the algorithm converges quicker but the performance on the test set is not very good. With bigger batch size, the algorithm converges slower, and the best result is obtained with batch_size = 64.

- *learning rate:*  
  we test learning rate = 0.0001, 0.001, 0.01. Small lr -> converge slowly, easily become overfitting; Big lr-> can not even converge, underfitting. The best lr tested is 0.001 for optimiser Adam, and 0.01 for SGD with momentum.

- *number of epochs:*  
  this term is almost depended on the two hyper parameters above. As in the provided training function, the parameters of the network which obtained the best performance on validation set is kept as the final network's parameters, the learning rate should then be big enough to get the smallest loss on validation set.

- *size of the training set/validation set:*  
  Obviously if we take more data as the train set, our model could get better performance, this is because training a NN is a data-based task. But we still need to keep enough data for the validation set to help us tune the hyper parameters(we can not do it directly with the test set because then the model may 'overfit' on the test set, and the results may not be the real performance of the model.) In this work, our strategy is: Firstly using 40000 datas for train set and 10000 datas for validation set -> Tuning the hyper parameters to train a model which could get the best performance on validation set -> Fix the hyper parameters, then using 49000 datas for train set and 1000 datas for validation set to retrain the model -> test the model obtained on the test set. 

### 2. What is happening when the training and test losses start diverging?
- When the training loss diverging: learning rate too big, the network can not converge.
- When test(or validation) loss diverging: overfitting.

Then we have to tune the hyper parameters to try to avoid these problems.

### 3. Top performing configurations:

**Network structure:** (we finally construct our network like a simplified VGG network)

[CNN + relu + CNN + relu + Maxpooling]x3 + FC + relu + dropout + FC + relu + FC (see more details in our notebook)

- *Number of convolutional layers:* 6

- *Kernel size for all CNN layers:* 3x3

- *Number of feature maps:* from 32 -> 64 -> 128 to 256

- *Number of Maxpooling layers:* 6

- *Number of Fully connected layers:* 3

- *Number of Fully connected layers:* from 256x4x4 -> 256 -> 64 to 10

- *Number of Dropout layers:* 1

**Hyper parameters:**

- *Learning rate:* 0.01

- *batch size:* 64

- *Number of epochs:* 20

- *size of the training set:* 49000

- *size of the validation set:* 1000

**Other configurations:**

- *Optimizer:* SGD with momentum = 0.9, lr = learning rate

- *Apply data augmentation for train and validation set by using:* transforms.RandomHorizontalFlip(), transforms.RandomGrayscale();

**Result: (best performance)**

*Accuracies:*

- Accuracy of the network on the 49000 train images: 95.12 %

- Accuracy of the network on the 1000 validation images: 83.30 %

- Accuracy of the network on the 10000 test images: 82.94 %
  
The result is much better compared to the starting CNN 

### 3.1 Interpretation on the losses plot and confusion matrix

- Plot of losses:

![loss](loss.png)

We see that the more the epochs increases, the more the curves try to cross each other until the validation curve exceeds the training curve. It is probably around this moment that the overlearning appears.



- Normalized confusion matrix:

![Confusion](confusion.png)

We can see that the biggest problem remains the photos of cats which are predicted like photos of dogs and vice versa.

### 3.2 Potential improvements

- Fix the overfitting problem
- Play on other parameters of the CNN : stride, padding, atrous (we did not focus on these parameters because the size of our image is relatively small).
- When we detect an image of cats or dogs, these images are redirected to a more suitable CNN algorithm. This second algo will have as input only photos of dogs and cats, so it will have to differentiate between the two species.
