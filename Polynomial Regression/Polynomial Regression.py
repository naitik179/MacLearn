# Polynomial Regression Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

dataset.head()

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[: , 2].values

# building Linear Regression model

from sklearn.linear_model import LinearRegression
line_reg = LinearRegression()
line_reg.fit(X , y)

# fitting ploynomial Regression to the dataset
# automatically includes the constant X0 as 1 
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

line_reg2 = LinearRegression()
line_reg2.fit(X_poly , y)

# visualizing the linear Regression model
plt.scatter(X , y , color = "red")
plt.plot(X , line_reg.predict(X) , color = "blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

# visualizing the Ployregression Model
X_grid = np.arange(min(X) , max(X) , 0.1)
X_grid = X_grid.reshape((len(X_grid)) , 1) 
plt.scatter(X , y , color = "red")
plt.plot(X_grid , line_reg2.predict(poly_reg.fit_transform(X_grid)) , color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("position")
plt.ylabel("Salary")
plt.show()

# predicting the salary in linear Regression model 
line_reg.predict([[6.5]])

# predicting the salary in Polynomial Regression model 

line_reg2.predict(poly_reg.fit_transform([[6.5]]))