# Lab2 Image Mining

By Jiangnan HUANG & Thomas SU - 21/10/2020

Our notebook [*Click here*](https://github.com/UniversalDependencies).
 
### Explain the neural networks and hyper parameters configurations we tested and the resulting performances and trade-offs found.



**neural networks:**

We test CNN based model with different structures.
- kernel size;
- number of feature maps;
- number of convolutional layers;
- size of fully connected layers;
- use dropout or not(also number of dropout layers);

**hyper parameters:**

- batch size: we test batch_size = 32, 64, 128, 256, 512. With small batch size(batch_size = 32), the algorithm converges quicker but the result on the test set is not very good. With bigger batch size, the algorithm converges slower, and the best result is obtained with batch_size = 64.
- learning rate: we test learning rate = 0.0001, 0.001, 0.01. Small lr -> converge slowly, easily become overfitting; Big lr-> can not even converge, underfitting. The best lr tested is 0.001.
- number of epochs: this term is almost depended on the two hyper parameters above. As in the provided training function, the parameters of the network which obtained the best performance on validation set is kept as the final network's parameters, the learning rate should then be big enough to get the smallest loss on validation set.
- the training set to validation set ratio:

The report should contain inside a link to your notebook saved into your github account:  In the colab notebook do:File→Save a copy in Github, and add that link to your repo in your report. Provide an interpretation on the losses plot for your top performing configurations and insights on potential improvements.  

### What is happening when the training and test losses start diverging?
- When the training loss diverging: learning rate too big.
- When test(or validation) loss diverging: overfitting.

All used parameters must be reported (at least learning rates, batch size, nr of epochs, accuracy on the image test set and plot of losses).  The best performance will be used for a ranked leader board.
