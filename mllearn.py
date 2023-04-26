from sklearn.linear_model import LinearRegression

# Define the data
X = [[0], [1], [2], [3]] # independent variable
y = [1, 2, 3, 4] # dependent variable

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make a prediction
prediction = model.predict([[4]])

print(f'Prediction: {prediction[0]}')