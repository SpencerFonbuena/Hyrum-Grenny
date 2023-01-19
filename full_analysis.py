import glob
import pandas as pd
import numpy as np
import csv


#these are the input needed to make the program run

#This is will give you the RSI indicator, and takes as input the lookback period you want for the RSI
#and the filepath of the data you would like analyzed
def rsi_divergence(lookbackperiod, filepath):
        #sets how much of the array you print out. Can be helpful when troubleshooting
    np.set_printoptions(threshold=1000)

    #imports the data and names it aaa_1hour
    aaa_1hour = pd.read_csv(f"{filepath}", header=None)

    #gives the data headers 
    headers = ("Date", "Open", "High", "Low", "Close", "Volume")
    aaa_1hour.columns = headers

    #calculated the difference between closing prices and stores it in close_difference
    close_difference = aaa_1hour['Close'].diff()

    #storing the average values
    average_loss = np.zeros(0)
    average_gain = np.zeros(0)

    #finally store the RSI vallues in an array
    rsi_indicator = np.zeros(0)


    #initializes your desired moving average and look back periods
    LOOK_BACK_PERIOD = lookbackperiod

    #store the counter range
    b = 0
    e = LOOK_BACK_PERIOD
    #increment through each range of 14 values
    for j in range(0, len(close_difference)):
        
        #create the subset of data that the next for loop operates on
        data_range = close_difference.iloc[b:e]
        
        #temporarily store the negative and positive values within the look back period
        loss_array = np.zeros(0)
        gain_array = np.zeros(0)

        #check to see if we are at the end of the dataset
        if e < len(close_difference):

            #seperate each of the datapoints into positive and negative
            for i in data_range:
                if i < 0:
                    loss_array = np.append(loss_array, i)
                if i > 0:
                    gain_array = np.append(gain_array, i)
        
            #calculate and send average loss and gain data into the variables average_loss and average_gain       
            average_loss = np.append(average_loss, np.sum(np.abs(loss_array))/(e-b))
            average_gain = np.append(average_gain, (np.sum(gain_array)/(e-b)))

            #increment the counter to move to the next set of 14 data points
            b += 1
            e += 1

    c = 0
    d = LOOK_BACK_PERIOD + 2

    #not entirely sure what this does if i'm being honest, but this is the welles wilder smoothing approach. This will essentailly get us our final RSI indicator value
    for k in range(0, len(average_gain)):
        if d < len(average_gain):
            if close_difference[d] >= 0:
                rsi_indicator = np.append(rsi_indicator, (100 - (100 / (1 + ((average_gain[c] * 13 + close_difference[d])/(average_loss[c] * 13))))))
            if close_difference[d] < 0:
                rsi_indicator = np.append(rsi_indicator, (100 - (100 / (1 + ((average_gain[c] * 13)/(average_loss[c] * 13 + np.abs(close_difference[d])))))))
            c += 1
            d += 1
    
    #this cuts of the last 30 values of the price close data to account for all of the moving averages (14 for the look back, and 16 for the wilder smoothing method)
    close_prices = aaa_1hour["Close"]

    return close_prices, rsi_indicator

#gives us the polynomial best fit line for both the RSI and the closing prices. returns two arrays, one of the best fit rsi line, and the other
#of the best fit closing price line, both using polynomial best fit methods
def polynomial_linefit(closing, relative_strength, degrees,):

    #this gives us the closing price data and the rsi data
    crazy = closing
    rsi = relative_strength

    #This gives us our x values to plot our closing price data against and our rsi data against
    x_values = range(0,len(crazy))
    x_values_rsi = range(0, len(rsi))

    #This gives the polynomial best fit line equation, inputs are (x values, the closing price data (y values), and the degree to which you want the polynomial)
    poly_line = np.polyfit(x_values, crazy, degrees)
    poly_line_rsi = np.polyfit(x_values_rsi, rsi, degrees)

    #this initializes and declares the array to store all of the data points in so we can plot them and visualize the data
    polynomial_points = np.zeros(0)
    rsi_points = np.zeros(0)
    
    #This increments in the for loop
    a = 0


    #increments through and creates array of best fit line points (with the closing price data) using a polynomial function
    for i in range(0, len(crazy)):
        #This gives you the y value, at an x(here we used a) input. Essentially, it allows for the plotting of the line
        poly_boi = np.polyval(poly_line, a)
        #This feeds data into the array
        polynomial_points = np.append(polynomial_points, poly_boi)
        a += 1

    #resets the counter
    a = 0

    #increments through and creates array of best fit line points (witht the RSI data) using a polynomial function
    for j in range(0, len(rsi)):
        #This gives you the y value, at an x(here we used a) input. Essentially, it allows for the plotting of the line
        rsi_poly_points = np.polyval(poly_line_rsi, a)
        #This feeds data into the array
        rsi_points = np.append(rsi_points, rsi_poly_points)
        a += 1
    return poly_line, poly_line_rsi, rsi_points, polynomial_points   

#returns two arrays. One array of derivative points linked to the RSI data, and one array of derivative points linked to the close price data
def polynomial_derivative(der_rsi_equation, der_close_equation, der_rsi_points, der_close_points):
    rsi = der_rsi_equation
    close = der_close_equation
    rsi_points = der_rsi_points
    close_points = der_close_points

    #derivative equation, spits out a bunch of coefficients
    rsi_equation = np.polyder(rsi)
    close_equation = np.polyder(close)

    #arrays that hold the derivative values, point by point as it goes through time
    rsi_line = np.zeros(0)
    close_line = np.zeros(0)

    a = 0

    for i in range(0, len(rsi_points)):
        intermediate_rsi = np.polyval(rsi_equation, a)
        rsi_line = np.append(rsi_line, intermediate_rsi)
        a += 1

    a = 0

    for i in range(0, len(close_points)):
        intermediate_close = np.polyval(close_equation, a)
        close_line = np.append(close_line, intermediate_close)
        a += 1

    return rsi_line, close_line

#Compares the rsi and the price slopes to find out when they are in agreeance, and when they disagree, and returns an array of that data. 
def comparison(rsi_final_fun, close_final_fun):
    #Assigning module inputs to variables
    rsi_line = rsi_final_fun
    close_line = close_final_fun
    
    
    bool_array = np.zeros(0)
    a = 0
    b = 0
    c = 0
    for i in range(0, len(rsi_line)):
        #output a 1 when the rsi and trend agree bullish
        if (rsi_line[c] > 0 and close_line[c] > 0):
            bool_decision = 1
            bool_array = np.append(bool_array, bool_decision)
        #output a 1 when the rsi and trend agree bearish
        elif (rsi_line[c] < 0 and close_line[c] < 0):
            bool_decision = 1
            bool_array = np.append(bool_array, bool_decision)
        #output a 3 when the rsi and trend disagree, and the rsi is bearish
        elif (rsi_line[c] < 0 and close_line[c] > 0):
            bool_decision = 3
            bool_array = np.append(bool_array, bool_decision)
        #output a 4 when the rsi and trend disagree, and the rsi is bullish
        else:
            bool_decision = 4
            bool_array = np.append(bool_array, bool_decision)
            b += 1
        c += 1
    return bool_array

def investment_analysis(price, boolean_array):
    #The array of 1,3, or a 4. 1 means there is not divergence, 3 means there is bearsih divergence, 4 means bullish divergence
    divergence = boolean_array
    #array of closing prices taken straight from the data
    closers = price

    #arrays to store the values of where divergence starts and stops
    decrease = np.zeros(0)    
    increase = np.zeros(0)


    #array to store whether or not the divergence was accurate. (0 if not accurate, 1 if accurate)
    outcome = np.zeros(0)
    a = 0

    #there is a minus 1 so that the loop will end once the iterations have completely finished. This loop outputs two arrays. The values are
    #where the divergence begins and ends.
    while a < len(divergence) - 1:

        #this if statement is here, because of the last decrease = np.append on line 214. It ensures that the last value is input correctly
        #to analyze whether or not the price followed through as it should have
        if divergence[a] == 3:
            #checks if the value is a 3, and then proceeds
            while divergence[a] == 3 and a < len(divergence) - 1:
                #the reason that this math is here, is that if i and i + 1 is 4, that means the divergence has ended, and it should exit
                if divergence[a] + divergence[a + 1] == 4 and a < (len(divergence) - 1):
                    decrease = np.append(decrease, a)
                a += 1
            while divergence[a] == 1 and a < (len(divergence) - 1):
                a += 1
            decrease = np.append(decrease, a)
        #this if statement is here, because of the last increase = np.append on line 214. It ensures that the last value is input correctly
        #to analyze whether or not the price followed through as it should have
        if divergence[a] == 4 and a < (len(divergence) - 1):
            while divergence[a] == 4 and a < (len(divergence) - 1):
                if divergence[a] + divergence[a + 1] == 5 and a < (len(divergence) - 1):
                    increase = np.append(increase, a)
                a += 1
            while divergence[a] == 1 and a < (len(divergence) - 1):
                a += 1
            increase = np.append(increase, a)
        while divergence[a] == 1 and a < (len(divergence) - 1):
            a += 1


    
    #There was a problem where the loop would just add the last index from the list. This gets rid of that error. 
    #logically, each increase and decrease should have a start and an end value, ergo, an even amount of values. 
    decrease = decrease[:len(decrease) - (len(decrease) % 2)]
    increase = increase[:len(increase) - (len(increase) % 2)]

    a = 0
    while a < len(decrease):
        if closers[decrease[a]] > closers[decrease[a] + 1]:
            outcome = np.append(outcome, True)
        else:
            outcome = np.append(outcome, False)
        a += 2
    
    #initialize counter variable again
    a = 0
    while a < len(increase):
        if closers[increase[a]] < closers[increase[a] + 1]:
            outcome = np.append(outcome, True)
        else:
            outcome = np.append(outcome, False)
        a += 2
    try:
        average = sum(outcome) / len(outcome)
    except ZeroDivisionError:
        average = .5
    number_length = len(outcome)
    print(average, number_length)
    return average, number_length

#this loop takes a folder, and inputs the files one by one into the code to be analyzed, and returns the probabilities into an array called outcome array
data_folder = glob.glob('/Users/spencerfonbuena/Documents/Trading/Trading_Datasets/Stocks Data/1 Hour/stocks-complete_tickers_A-B_1hour_1cd943/*.txt')
print(data_folder)
#two arrays storing final array data for the percentage of time that the divergence held true
outcome_array = np.zeros(0)
#array that holds the number of observations from each file
n_length = np.zeros(0)

#cycle through each file in a trading dataset folder
for counter, files in enumerate(data_folder, 1032):
    try:
        data_file = data_folder[counter]

        look_back_period = 14
        #this assigns variable names to the return values of the function (rsi_divergence)
        close_prices, rsi_indicator = rsi_divergence(look_back_period, data_file)
        if len(close_prices) > 40:
            poly_degree = 25
            #this calls the function "polynomial_linefit" and passes into it the close prices parsed from the rsi_divergence function, and the 
            #user input poly_degree, which is the degree the user wants to use to fit the line
            derivative_rsi_equation, derivative_closing_equation, derivative_rsi_points, derivative_close_points = polynomial_linefit(close_prices, 
                rsi_indicator, poly_degree)

            #returns two arrays. One array of derivative points linked to the RSI data, and one array of derivative points linked to the close price data
            rsi_final, close_final = polynomial_derivative(derivative_rsi_equation, derivative_closing_equation, derivative_rsi_points, 
                derivative_close_points)

            divergent = comparison(rsi_final, close_final)

            average_array, n_array = investment_analysis(close_prices ,divergent)

            outcome_array = np.append(outcome_array, average_array)
            n_length = np.append(n_length, n_array)
            print(counter)
    except IndexError:
        break
        
#the total number of data points. Used in order to get an overall historical probability
total_length = sum(n_length)
semifinal_value = np.zeros(0)

#Weight all the averages and give a final value
for row, value in enumerate(outcome_array):
    percent_length = n_length[row] / total_length
    semifinal_value = np.append(semifinal_value, outcome_array[row] * percent_length)

final_value = sum(semifinal_value)
print(final_value)
print(semifinal_value)
print(outcome_array)