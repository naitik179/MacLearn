# implementing Descision Tree Regressor model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[: , 2].values

# y = y.reshape(len(y) , 1)

# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# sc_y = StandardScaler()
# X = sc_x.fit_transform(X)
# y = sc_y.fit_transform(y)

# building the Regressor Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# predicitng the new Result
y_pred = regressor.predict([[6.5]])

# visualizing the model for smother resolution

X_grid = np.arange(min(X) , max(X) , 0.1)
X_grid = X_grid.reshape(len(X_grid) , 1)
plt.scatter(X , y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Descision Tree Regressor)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


