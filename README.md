# **DeepLearning**
This contains the codes of deep neural network on different datasets using tensorflow.

## **tensorflow**

### neural_net_mnist.py
In this code we have made a deep neural network having three hidden layers using tensorflow.After this we train our
model on mnist dataset which contains images of numbers(0-9) of 28*28 pixels.We got accuracy of ~95%.

### convolution_neural_net_mnist.py
In this code we have trained the convolution neural network on mnist dataset and got the accuracy ~97%.

### recurrent_neural_net_mnist.py
In this code we have trained the convolution neural network on mnist dataset and got the accuracy ~98.5%.

### create_sentiment_featuresets.py
In this code we deal with the realistic dataset.In this we are converting the data into vectors which we can train in our neural network made in mnist_tensorflow.py .

### sentiment_neural_network.py
In this code we use the vectors made in create_sentiment_featuresets.py and neural network of neural_net_mnist.py.In this we got accuracy of ~68%. It has so low accuracy because of very less data.It can achieve high accuracy if we have larger dataset. 

### large-data/sentiment_analysis.py
In this code we have done sentiment analysis on realistic data which is quite large using deep neural network.

## **tflearn**
This contains the codes of deep neural network on different datasets using tflearn.

### tflearn_convnet_mnist.py
In this code we have trained the convolution neural network on mnist dataset and got the accuracy ~98%.

It contains some codes of tutorial by sentdex.
