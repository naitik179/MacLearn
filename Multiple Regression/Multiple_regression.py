import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[: , 4].values

# handling categorical data by encoding independednt variable 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# LabelEncoder is not required in the latest version of sklearn 
# also dont to the .toarray
#labelencoder_X = LabelEncoder()
# X[:,3] = labelencoder_X.fit_transform(X[: , 3])


onehotencoder =  ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = onehotencoder.fit_transform(X)
X = X.astype('float64')


# Avoiding the dummy variable trap 
X = X[: , 1:]

# splitting the data 

from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split(X ,y, test_size =0.2 ,random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train) 

#Predicitng the test set
y_pred = regressor.predict(X_test)

# building the optimal model using Bakward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50 , 1)).astype(int) , values = X , axis = 1)


# Step 2 fit the full model with all possible predictors
X_opt = X[: , [0 ,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

# consider the predictor with the highest P value greater than SL . if P> SL then remove the predictor 

X_opt = X[: , [0 ,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[: , [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,3,5]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[: , [0,3]]
regressor_OLS = sm.OLS(endog = y , exog = X_opt).fit()
regressor_OLS.summary()













