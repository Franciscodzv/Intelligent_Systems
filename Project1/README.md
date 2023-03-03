# Project 1 - Linear Regression
This project uses Linear regression to predict how popular a song can be based on some of its features. The data I used is from: https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre 

The data contains several columns, but I only used the following columns: Energy, Danceability, and Instrumentalness. I also used the column "Popularity" as the target variable. I used these columns because I want to know how much the actual music and vibe of the song affects its popularity.

As of this project I was only able to get my coefficient of determination is 0.45. This is a good start, however, for making more precise predictions I would need to either use a different model or to handle the data and "clean" it before making the predictions. I will do this for another handin and I will compare and contrast the results.



## How to run the code
Whenever you want to run the code, you need to make sure that you have the following packages installed: pandas, numpy, matplotlib, sklearn and math. 

Once you have installed these packages, you can run the code by running the file "project1.py" in the folder "Project1".

Once the code is running, you will be prompted to enter the values of the features we're using to predict the popularity of the song. The values you enter should be between 0 and 1. The only feature that is not numerical is the music genre, for this feature you will enter a String. The prompt specifies which feature you are entering the value for.

Examples:

Danceability: 0.5
energy: 0.7
instrumentalness: 0.25
Genre: Rock

### Make another prediction?

Once you have entered the values, the code will print out the predicted popularity of the song. Then it will plot data regarding the prediction and the features. For example, it will plot the predicted values vs the real values. It will also plot each feature vs the popularity, this is done for analysis purposes.

After this you will be asked if you want to make a new prediction. If you enter "y" || "Y" || "Yes" || "yes" the code will run again, if you enter "n" || "N" || "No" || "no" the code will stop running.


## Analysis
Let's run the code and enter the following values:

Danceability: 0.50

Energy: 0.75

Instrumentalness: 0.33

Genre: Alternative

After running the code, we get the following popularity prediction: 32.38 This means that the song has a 32.38% chance of being popular. But if my coefficient of determination is of 0.45 then can we trust this prediction?

The answer is no. 

Let's look at this plot:
![alt text](https://github.com/Franciscodzv/Intelligent_Systems/blob/master/Project1/prediction_vs_actual.png "Prediction vs Actual")

From the predictions, we can see that while some of the values are accurate, others are not. This indicates that the model may be overfitting, which implies that it is performing well on the training data but struggles when presented with new data.

Since my data contains thousands of rows there might be noisy data feeding my model irrelevant information. 

I tried adding more features to my model, but this didn't help. I also tried to remove some of the features, but this didn't help either. 


