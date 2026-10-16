import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


boston_house = fetch_openml(
    name = "boston",
    version=1,
    as_frame=False

)

n = False

# getting information:
if n:
    print(boston_house.keys())
    print(boston_house.feature_names)
    print(boston_house.data)
    print(boston_house.DESCR)

# make Variable for important things:
boston_data = boston_house.data # features
boston_target = boston_house.target # price

# testing place:
# print(np.shape(boston_data))


# Selecting the desired feature:
boston_data = boston_data
# testing place: 
# print(np.shape(boston_data))


x_train, x_test, y_train, y_test  = train_test_split(boston_data, boston_target)

# testing place:
print(np.shape(x_train))
print(np.shape(y_train))

# make a chart:
# plt.scatter(x_train,y_train)
# plt.xlabel("rooms")
# plt.ylabel("price")
# plt.show()

# teaching the model:
# regressor = LinearRegression()
regressor = Lasso(alpha=0.01)

regressor.fit(x_train,y_train)

regressor.predict(x_test)
print(regressor.score(x_test,y_test))
print(regressor.score(x_train,y_train))