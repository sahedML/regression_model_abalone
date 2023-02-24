
# develop and uses of machine learning model "regression" and model optimization
# data = abalone dataset to determine the age
# data source: UCI machine learning repository
# Sahed Ahmed Palash
# August 2022
# Dhaka, Bangladesh

# 1. predicting the age of abalone using LinearRegression model
# import important packages
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor

# import the abalone dataset
abalone = pd.read_csv("abalone.data")

# create column name
abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole Weight",
                   "Shucked Weight", "Viscera Weight", "Shell Weight", "Rings"]
# print(abalone.head())

# data exploration
# we can check the distribution of our dependant or the target variable using histogram
# abalone["Rings"].hist(bins=15)
# plt.show()

# let's see the correlation matrix to determine which features
# correlation_matrix = abalone.corr()
# print(correlation_matrix["Rings"])  # weight has the highest correlation with the rings
abalone = abalone.drop(["Sex"], axis=1)

# create x and y variables
x = abalone.drop("Rings", axis=1)
x = x.values
y = abalone["Rings"]
y = y.values

# let's split the data set into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# create a model variable called linear regression
model = LinearRegression()

# now train the model with x and y data
model.fit(x_train, y_train)

# print the model result
# print("the intercept of the model is: ", model.intercept_)
# print("the slope of the model is: ", model.coef_)
# print the r squared value
r_sq = model.score(x, y)
# print("the coefficient of the model is: ", r_sq)  # model accuracy is 73% which is good

# predicting the test data using the pre-trained model
y_predict = model.predict(x_test)
mse_predict = mean_squared_error(y_test, y_predict)
rmse_train = sqrt(mse_predict)
print("rmse_regression: ", rmse_train)
print("regression_predict: ", y_predict[0:10])
print("regression_actual: ", y_test[0:10])

# plotting the model data
# A last thing to look at before starting to improve the model is the actual fit of our model.
# To understand what the model has learned, we can visualize how our predictions have been made
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(x_test[:, 0], x_test[:, 1], c=y_predict, s=50, cmap=cmap)
f.colorbar(points)
plt.title("prediction against training data")
plt.show()

# To confirm whether this trend exists in actual abalone data,
# we can do the same for the actual values by simply replacing the variable
# that is used for c

cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=50, cmap=cmap)
f.colorbar(points)
plt.title("prediction against actual data")
plt.show()
# we can also do the same with the other independent variables as well
print("----------------------------------------------------------------------------------------")

# let's use other regressors to see if we can improve the performance of our model
# 2. KNN regressor
# develop a KNN model and fit the data into it and evaluate the model
KNN_Model = KNeighborsRegressor(n_neighbors=3)
KNN_Model.fit(x_train, y_train)
model_prediction = KNN_Model.predict(x_train)
mse_KNN_train = mean_squared_error(y_train, model_prediction)
rmse_KNN_train = sqrt(mse_KNN_train)
# print("rmse_KNN_train: ", rmse_KNN_train)

# working with test data
# now lets check the rmse of our test data and compare with the rmse of our training data
# rmse of training and test data should be as close as possible, if not then go for optimization
KNN_predict = KNN_Model.predict(x_test)
mse_KNN_test = mean_squared_error(y_test, KNN_predict)
rmse_KNN_test = sqrt(mse_KNN_test)
print("rmse_KNN_test: ", rmse_KNN_test)
print("KNN_predict: ", KNN_predict[:10])
print("KNN_actual: ", y_test[:10])
print("----------------------------------------------------------------------------------------")

# improving model performance using GridSearchCV
parameters = {"n_neighbors": range(1, 50)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(x_train, y_train)

# in the end, it will retain the best performing value of k,
# which you can access with .best_params_:
# print(gridsearch.best_params_)  # best value for K = 29

# now we will try this value to our existing gridsearch model and see its performance
gridSearch_train_predict = gridsearch.predict(x_train)
mse_grid_train = mean_squared_error(y_train, gridSearch_train_predict)
rmse_grid_train = sqrt(mse_grid_train)
# print("rmse_grid_train: ", rmse_grid_train)

# same with the test data
gridSearch_test_predict = gridsearch.predict(x_test)
mse_grid_test = mean_squared_error(y_test, gridSearch_test_predict)
rmse_grid_test = sqrt(mse_grid_test)
print("rmse_grid_test: ", rmse_grid_test)
print("gridSearch_prediction: ", gridSearch_test_predict)
print("gridSearch_actual: ", y_test[0:10])
print("----------------------------------------------------------------------------------------")

# improving model performance using weighted average of neighbors in GridSearchCV
parameter = {
    "n_neighbors": range(1, 50),
    "weights": ["uniform", "distance"]
}
gridSearch_weighted = GridSearchCV(KNeighborsRegressor(), parameter)
gridSearch_weighted.fit(x_train, y_train)
# print(gridSearch_weighted.best_params_)  # {'n_neighbors': 31, 'weights': 'distance'}
gridSearch_weighted_test = gridSearch_weighted.predict(x_test)
mse_grid_weighted = mean_squared_error(y_test, gridSearch_weighted_test)
rmse_grid_weighted = sqrt(mse_grid_weighted)
print("rmse_grid_weighted: ", rmse_grid_weighted)
print("gridSearchWeighted_predict: ", gridSearch_weighted_test[0:10])
print("gridSearchWeighted_actual: ", y_test[0:10])
print("----------------------------------------------------------------------------------------")

# improving model performance using Bagging
# First, create the KNeighborsRegressor with the best choices for k and weights that you got from GridSearchCV
best_k = gridSearch_weighted.best_params_["n_neighbors"]
best_weight = gridSearch_weighted.best_params_["weights"]
bagged_KNN = KNeighborsRegressor(n_neighbors=best_k, weights=best_weight)

# create a new instance with 100 estimators using the bagged_knn model
bagging_model = BaggingRegressor(bagged_KNN, n_estimators=100)
bagging_model.fit(x_train, y_train)

# Now we can make a prediction and calculate the RMSE to see if it improved
bagging_prediction = bagging_model.predict(x_test)
mse_bagging = mean_squared_error(y_test, bagging_prediction)
rmse_bagging = sqrt(mse_bagging)
print("rmse_bagging: ", rmse_bagging)
print("bagging_prediction: ", bagging_prediction[:10])
print("bagging_actual: ", y_test[:10])
