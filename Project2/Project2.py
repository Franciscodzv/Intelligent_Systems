import numpy as np #numpy is used to make some operrations with arrays more easily
import time
import random as rand




__errors__= [];  #global variable to store the errors/loss for visualisation

def h(params, sample):
	"""This evaluates a generic linear function h(x) with current parameters.  h stands for hypothesis

	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		sample (lst) a list containing the values of a sample 

	Returns:
		Evaluation of h(x)
	"""
	acum = 0
	for i in range(len(params)):
		acum = acum + params[i]*sample[i]  #evaluates h(x) = a+bx1+cx2+ ... nxn.. 
	return acum;

def show_errors(params, samples,y):
	enter = 0
	global __errors__
	error_acum =0
	for i in range(len(samples)):
		hyp = h(params,samples[i])
		print( "hypothesis:  %f  y: %f " % (hyp,  y[i]))   
		#time.sleep(0.00005)
		error = hyp - y[i]
		error_acum = error_acum + error**2 # this error is the original cost function, (the one used to make updates in GD is the derivated verssion of this formula)
	mean_error_param= error_acum / len(samples)
	__errors__.append(mean_error_param)



def r_squared(y_true, y_pred):
    """Calculate R-squared given the true and predicted values"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = sum((y_true - y_pred) ** 2)
    ss_tot = sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return abs(r2)


##get the mean squared error
def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error given the true and predicted values"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = sum((y_true - y_pred) ** 2) / len(y_true)
    return mse






def GD(params, samples, y, alfa):
	"""Gradient Descent algorithm 
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		samples (lst) a 2 dimensional list containing the input samples 
		y (lst) a list containing the corresponding real result for each sample
		alfa(float) the learning rate
	Returns:
		temp(lst) a list with the new values for the parameters after 1 run of the sample set
	"""
	temp = list(params)
	for j in range( len(params) ):
		acum = 0 
		for i in range(len(samples)):
			error = h( params, samples[i] ) - y[i]
			acum = acum + error * samples[i][j]  #Sumatory part of the Gradient Descent formula for linear Regression.
		temp[j] = params[j] - alfa * ( 1 / len(samples) ) * acum  #Subtraction of original parameter value with learning rate included.
	return temp

samples = [[],[],[]]
params = [0,0,0,0]



import csv

with open('/Users/pacodiaz/Documents/Intelligent Systems/Intelligent_Systems/Project2/gas_prices.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)

    # Read all the rows of the CSV file into a list
    rows = list(reader)

    # Shuffle the rows randomly
    rand.shuffle(rows)

# Open the same CSV file for writing and create a CSV writer object
with open('/Users/pacodiaz/Documents/Intelligent Systems/Intelligent_Systems/Project2/gas_prices.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the shuffled rows back to the CSV file
    writer.writerows(rows)

with open('/Users/pacodiaz/Documents/Intelligent Systems/Intelligent_Systems/Project2/gas_prices.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the header row

    
    latitudes = []
    longitudes = []
    years = []
    pricesY = []
    original_samples = []

    latitudes_test = []
    longitudes_test = []
    years_test = []
    pricesY_test = []
    original_samples_test = []
    for row in reader:
	    
        if row[3] == '' or row[4] == '' or row[5] == '':
            continue
            
        if rand.random() < 0.8:
            latitudes.append(float(row[3]))
            longitudes.append(float(row[4]))
            year = row[2]
            year = int(year[-4:])
            years.append(year)
            pricesY.append(float(row[5])) #price
            original_samples.append([1,row[3],row[4],year])
        else:
            latitudes_test.append(float(row[3]))
            longitudes_test.append(float(row[4]))
            year_test = row[2]
            year_test = int(year_test[-4:])
            years_test.append(year_test)
            pricesY_test.append(float(row[5]))
            original_samples_test.append([1,row[3],row[4],year_test])


    # Apply min-max normalization to the latitudes
    
    min_lat = min(latitudes)
    max_lat = max(latitudes)
    normalized_latitudes = [(lat - min_lat) / (max_lat - min_lat) for lat in latitudes]   

    min_long = min(longitudes)
    max_long = max(longitudes)
    normalized_longitudes = [(long - min_long) / (max_long - min_long) for long in longitudes]

    # Apply min-max normalization to the years
    min_years = min(years)
    max_years = max(years)
    normalized_years = [(year - min_years) / (max_years - min_years) for year in years]

    ##apply normalized values to the test set
    min_lat_test = min(latitudes_test)
    max_lat_test = max(latitudes_test)
    normalized_latitudes_test = [(lat - min_lat_test) / (max_lat_test - min_lat_test) for lat in latitudes_test]

    min_long_test = min(longitudes_test)
    max_long_test = max(longitudes_test)
    normalized_longitudes_test = [(long - min_long_test) / (max_long_test - min_long_test) for long in longitudes_test]

    # Apply min-max normalization to the years
    min_years_test = min(years_test)
    max_years_test = max(years_test)
    normalized_years_test = [(year - min_years_test) / (max_years_test - min_years_test) for year in years_test]


    params = [0,0,0,0]
    samples = []
    test_samples = []
   
    ##append the normalized values to the samples matrix
    for i in range(len(normalized_latitudes)):
        samples.append([normalized_latitudes[i], normalized_longitudes[i], normalized_years[i]])
	
    for i in range(len(normalized_latitudes_test)):
        test_samples.append([normalized_latitudes_test[i], normalized_longitudes_test[i], normalized_years_test[i]])


#  multivariate example
#params = [0,0,0]
#samples = [[1,1],[2,2],[3,3],[4,4],[5,5],[2,2],[3,3],[4,4]]
#y = [2,4,6,8,10,2,5.5,16]

alfa =.025 #  learning rate
for i in range(len(samples)):
	if isinstance(samples[i], list):
		samples[i]=  [1]+samples[i]
	else:
		samples[i]=  [1,samples[i]]
print ("original samples:")
print (original_samples[20])
#samples = scaling(samples)
print ("scaled samples:")
print (samples[20])

input("Press Enter to continue...")


epochs = 0
final_params = []

while True:  #  run gradient descent until local minima is reached
	oldparams = list(params)
	print (params)
	params = GD(params, samples,pricesY,alfa)	
	show_errors(params, samples, pricesY)  #only used to show errors, it is not used in calculation
	print (params)
	epochs = epochs + 1
	if(oldparams == params or epochs == 250):   #  local minima is found when there is no further improvement
		print ("samples:")
		print(samples)
		print ("final params:")
		print(params)
		final_params = params
		break

import matplotlib.pyplot as plt  #use this to generate a graph of the errors/loss so we can see whats going on (diagnostics)
plt.plot(__errors__)
plt.show()



predicted_prices_test = []

for i in range(len(latitudes_test)):
	scaled_lat_test = (latitudes_test[i] - min_lat_test) / (max_lat_test - min_lat_test)
	scaled_long_test = (longitudes_test[i] - min_long_test) / (max_long_test - min_long_test)
	scaled_year_test = (years_test[i] - min_years_test) / (max_years_test - min_years_test)
	predictedY_test = final_params[0] + final_params[1]*scaled_lat_test + final_params[2]*scaled_long_test + final_params[3]*scaled_year_test
	predicted_prices_test.append(predictedY_test)
	print("Predicted price with test set: ", predictedY_test, "Actual price: ", pricesY_test[i])
	



predicted_prices_train = []

for i in range(len(latitudes)):
    scaled_lat = (latitudes[i] - min_lat)/(max_lat - min_lat)
    scaled_long = (longitudes[i] - min_long) / (max_long - min_long)
    scaled_year = (years[i] - min_years) / (max_years - min_years)
    predictedY = final_params[0] + final_params[1]*scaled_lat + final_params[2]*scaled_long + final_params[3]*scaled_year
    predicted_prices_train.append(predictedY)
    print("Predicted price with training set: ", predictedY, "Actual price: ", pricesY[i])

print("R squared test set: ", r_squared(pricesY_test, predicted_prices_test))
print("R squared train set: ", r_squared(pricesY, predicted_prices_train))
print("R squared difference:", r_squared(pricesY_test, predicted_prices_test) - r_squared(pricesY, predicted_prices_train))
print("MSE test error: ",mean_squared_error(predicted_prices_test, pricesY_test))
print("MSE train error: ",mean_squared_error(predicted_prices_train, pricesY))
    


user_input = input("Do you want to predict a price? (y/n) ")




while user_input == 'y' or user_input == 'Y':
	scaled_lat = (float(input("Enter latitude: ")) - min_lat) / (max_lat - min_lat)
	scaled_long = (float(input("Enter longitude: ")) - min_long) / (max_long - min_long)
	scaled_year = (float(input("Enter year: ")) - min_years) / (max_years - min_years)
	predictedY = final_params[0] + final_params[1]*scaled_lat + final_params[2]*scaled_long + final_params[3]*scaled_year
	predicted_prices_train.append(predictedY)
	print("Predicted price: ", predictedY)
	user_input = input("Do you want to predict another price? (y/n) ")
	
print("Thank you for using our gasoline price predictor!")


	