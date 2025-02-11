import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import missingno as msn

# Load the dataset (make sure the path is correct)
df = pd.read_csv("DATA\salary_experience_data.csv")  # Adjust path if necessary
df.head()

# Check shape, info, and null values
print(f'The shape of the dataset is {df.shape}')
df.info()
print(f'Missing values: \n{df.isnull().sum()}')

# Visualize missing data (if any)
msn.matrix(df) 

# Descriptive statistics
df.describe()

# Scatter plot to visualize relationship
plt.scatter(df['YearsExperience'], df['Salary'], color='black')
plt.xlabel("Years of experience")
plt.ylabel("Salary (in thousand)")
plt.title("Salary vs Years of Experience")
plt.show()

# Split data into features (X) and target (Y)
X = df[['YearsExperience']]
Y = df['Salary']

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Initialize the model and fit it to the training data
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# Predict on the training and test data
Y_train_pred = lin_reg.predict(X_train)
Y_test_pred = lin_reg.predict(X_test)

# Plotting the regression line with the training data
plt.scatter(X, Y, color='black', label='Data Points')
plt.plot(X_train, Y_train_pred, color='blue', label='Regression Line')
plt.xlabel('Years of experience')
plt.ylabel('Salary (in thousand)')
plt.title('Salary Prediction using Linear Regression')
plt.legend()
plt.show()

# Evaluate the model
print(f'The value of MAE is {mean_absolute_error(Y_test, Y_test_pred)}')
print(f'The value of MSE is {mean_squared_error(Y_test, Y_test_pred)}')
print(f'The value of RMSE is {np.sqrt(mean_squared_error(Y_test, Y_test_pred))}')
print(f"R_2 score is {r2_score(Y_test, Y_test_pred)}")

# Print out the coefficients (slope and intercept)
β1 = lin_reg.coef_[0]
print(f'The value of slope is {β1}')
β0 = lin_reg.intercept_
print(f'The value of intercept is {β0}')
