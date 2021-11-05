#Linear Regression Multivarient

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
df = pd.read_csv("house_data.csv")

featureNames = ["xNode", "grade", "bathrooms", "lat", "sqft_living", "view"]

msk = np.random.rand(len(df)) < 0.9
train = df[msk]
test = df[~msk]

x = train[["grade", "bathrooms", "lat", "sqft_living", "view"]] 
x.insert(0, "xNode", np.ones(x.shape[0]), True)
x.reset_index(inplace=True)
del x["index"]

#Data normalization
#temp = (x["sqft_living"] - np.mean(x["sqft_living"])) / (np.amax(x["sqft_living"]) - np.amin(x["sqft_living"]))
#del x["sqft_living"]
#x.insert(4, "sqft_living", temp, True)

y = train["price"].tolist()

theta = [1,1,1,1,1,1]

xTest = test[["grade", "bathrooms", "lat", "sqft_living", "view"]] 
xTest.insert(0, "xNode", np.ones(xTest.shape[0]), True)
yTest = test["price"].tolist()

m = x.shape[0]
n = x.shape[1]

def hypothesis(theta, x):
    return np.dot(x, np.transpose(theta))

# Mean square error
def cost(theta, x, y):
    result = 0
    h = hypothesis(theta, x)
    for i in range(m):
        result += (h[i] - y[i]) ** 2
    return (1 / (2 * m)) * result
    
# Cost function in gradient descent
def J(theta, x, y, feature):
    result = 0
    h = hypothesis(theta, x)
    for i in range(m):
        result += (h[i] - y[i]) * x[featureNames[feature]][i]
    return result

costIterations = []

def gradientDescent(theta, x, y, iterations, alpha):

    for iteration in range (iterations):
        tempTheta = []
        
        for i in range (n):
            temp = theta[i] - (alpha / m) * J(theta, x, y, i)
            tempTheta.append(temp)
        
        costIterations.append(cost(theta, x, y))
        # Simultaneously change theta
        theta = tempTheta
    
    print("Cost: ", costIterations)
    
    # Plot cost
    plt.figure(figsize=(10,6))
    plt.title("Cost")
    plt.plot(range(0, iterations), costIterations)
    plt.grid(True) #Always plot.grid true!
    plt.show()
    return theta
    
# Start Gradient Descent
theta = gradientDescent(theta, x, y, 150, 0.00000001)


plt.figure(figsize=(10,6))
plt.title("Test data")

plt.plot(xTest["lat"], yTest, 'o', markersize=5, label = 'Y Original', color = 'blue', alpha = 0.4)
plt.plot(xTest["lat"], hypothesis(theta, xTest), 'x', markersize = 5, label = 'Y Predicted', color = 'red')
plt.grid(True) #Always plot.grid true!
plt.xlabel("X Test - latitude")
plt.ylabel("Price ( Millions )")
plt.legend()
plt.show()

#Predicting new data
yPredicted = hypothesis(theta, xTest)
testError = np.sum(((yPredicted - yTest) ** 2)) / (len(yTest) * 2 )
print('Test Error = ', testError)


