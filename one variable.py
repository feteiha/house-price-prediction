# Linear regression with one variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
df = pd.read_csv("house_data.csv",index_col=0)

msk = np.random.rand(len(df)) < 0.9
train = df[msk]
test = df[~msk]

x = train["sqft_living"].tolist()
y = train["price"].tolist()

xTest = test["sqft_living"].tolist()
yTest = test["price"].tolist()

x = np.array(x)
y = np.array(y)

xTest = np.array(xTest)
yTest = np.array(yTest)
# Data normalization
#x = (x - np.mean(x)) / (np.amax(x) - np.amin(x))
#y = (y - np.mean(y)) / (np.amax(y) - np.amin(y))

# Learning rate
alpha = 0.00000001

m = len(x)

def hypothesis(theta0, theta1, xi):
    return theta0 + theta1 * xi

# Mean square error
def mse(theta0, theta1):
    result = 0
    for i in range (m):
        result = result + (hypothesis(theta0, theta1, x[i]) - y[i]) ** 2 
    return result / (m * 2)
    
# Cost function in Gradient descent
def J(theta0, theta1, first):
    result = 0 # 1 / (2 * len(x))
    for i in range (m):
        if first:
            result = result + (hypothesis(theta0, theta1, x[i]) - y[i]) 
        else:
            result = result + ((hypothesis(theta0, theta1, x[i]) - y[i]) * x[i])
    
    return result

cost = []

def GradientDescent(theta0, theta1, iterations): 
    i = iterations
    
    while i != 0:    
        # Update Theta simultaneously
        tmpTheta0 = theta0 - ( (alpha/m) * J(theta0, theta1, True ) )
        tmpTheta1 = theta1 - ( (alpha/m) * J(theta0, theta1, False) )
        theta0 = tmpTheta0
        theta1 = tmpTheta1
        
        cost.append(mse(theta0, theta1))
        i -= 1
    
    #Plot Final iteration
    plt.figure(figsize=(10,6))
    plt.title("Final Iteration")
    plt.plot(x, y,'rx',markersize=5)
    plt.plot(x[:], hypothesis(theta0, theta1, x), 'b-')
    plt.grid(True) #Always plot.grid true!
    plt.xlabel("X - Sqft_Lirving")
    plt.ylabel("Y - Price ( Millions )")
    plt.legend()
    plt.show()
        
    print("Cost: ", cost)
    
    plt.figure(figsize=(10,6))
    plt.title("Cost")
    
    plt.plot(range(0, iterations), cost)
    plt.grid(True)
    plt.show()
    return theta0, theta1
    
    
theta0 = 0
theta1 = 0


plt.figure(figsize=(10,6))

plt.title("Original data")

plt.plot(x, y,'rx',markersize=5)
plt.plot(x[:], theta0 + theta1 * x[:], 'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(theta0,theta1))
plt.grid(True) #Always plot.grid true!
plt.xlabel("X - Sqft_Lirving")
plt.ylabel("Y - Price ( Millions )")
plt.legend()
plt.show()

#Start gradient descent
theta0, theta1 = GradientDescent(theta0, theta1, 150)



plt.figure(figsize=(10,6))
plt.title("Test data")

plt.plot(xTest[:], yTest, 'o', markersize=5, label = 'Y Original', color = 'blue', alpha = 0.4)
plt.plot(xTest, theta0 + theta1 * xTest[:], 'x', markersize = 5, label = 'Y Predicted', color = 'red')
plt.grid(True) #Always plot.grid true!
plt.xlabel("X Test - Sqft_Lirving")
plt.ylabel("Price ( Millions )")
plt.legend()
plt.show()

#Predicting new data
yPredicted = theta0 + theta1 * xTest[:]
testError = np.sum(((yPredicted - yTest) ** 2)) / (len(yTest) * 2 )
print('Test Error = ', testError)
