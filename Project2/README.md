# Project 2 - Gasoline Price Prediction in Mexico

This proyect implements a gradient descent algorithm to perform a linear regression on a dataset of gasoline prices in Mexico. The dataset includes the price of gasoline, latitude and longitude, and year of a gas station. 

The program preprocesses the dataset by normalizing the input features and splitting it into training and test sets. It then trains a linear regression model on the training set and evaluates it on the test set. Finally, it plots the evolution of the cost function during training and computes the $R^2$ and mean squared error (MSE) of the trained model on the test set and training sets.

The data set was obtained from the following link: https://www.kaggle.com/datasets/juanagsolano/gas-stations-prices-for-mexico?resource=download 

## How to run the program
Whenever you need to run the program you must make sure you have the following packages: numpy, random, csv and matplotlib.

Once you have the packages installed, you can run the program by typing the following command in the terminal:

```python Project2.py```

## How to use the program
The program will first print an example of the original samples of the dataset followed by an example of the normalized samples of the dataset. 

To continue you must press enter.

After you press enter the program will start training the model. It will print the evolution of the cost function during training and the final value of the cost function. Included in this output are the the values of R^2 and MSE of the trained model on the test and training sets.

## How to make predictions

Once all the information has been printed the program will ask you if you want to make a prediction. You will need to type 'y' or 'Y' to make a prediction. If you type anything else the program will end.

If you decided to make a prediction you will need to type the latitude, longitude and year of the gas station you want to predict the price of. The program will then print the predicted price of the gas station.

### Example of a prediction

``` Do you want to predict a price? (y/n) y```

``` Enter the latitude: 19.51 ```

``` Enter the longitude: -99.13 ```

``` Enter the year: 2030 ```

``` The predicted price is: 21.86 ```


