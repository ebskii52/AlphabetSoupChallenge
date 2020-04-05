# AlphabetSoupChallenge

How many neurons and layers did you select for your neural network model? Why?

For our input layer, we must add the number of input features equal to the number of variables in our feature DataFrame.
In our hidden layers, our deep learning model structure will be slightly different—we’ll add two hidden layers with only a few neurons in each layer. To create the second hidden layer, we’ll add another Keras Dense class while defining our model. All of our hidden layers will use the relu activation function to identify nonlinear characteristics from the input values.
In the output layer, we’ll use the same parameters from our basic neural network including the sigmoid activation function. The sigmoid activation function will help us predict the probability that an employee is at risk for attrition.

Looking at our model summary, we can see that the number of weight parameters (weight coefficients) for each layer equals the number of input values times the number of neurons plus a bias term for each neuron. Our first layer has 55 input values, and multiplied by the eight neurons (plus eight bias terms for each neuron) gives us a total of 448 weight parameters—plenty of opportunities for our model to find trends in the dataset

Were you able to achieve the target model performance? What steps did you take to try and increase model performance?

NO.

increase EPOCH Runs from 100 to 1000 and also increase the Neural network from one hidden layer to two hidden layers. 



If you were to implement a different model to solve this classification problem, which would you choose? Why?

Although their predictive performance was comparable, their implementation and training times were not—the random forest classifier was able to train on the large dataset and predict values in seconds, while the deep learning model required a couple minutes to train on the tens of thousands of data points. In other words, the random forest model is able to achieve comparable predictive accuracy on large tabular data with less code and faster performance. The ultimate decision of whether to use a random forest versus a neural network comes down to preference. However, if your dataset is tabular, random forest is a great place to start.
