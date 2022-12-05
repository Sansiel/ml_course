
import pandas as pd
import numpy as np
# import operator
from sklearn import linear_model,metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv('C:\projects\ml_course-main\Lab 3\communities.data', header=None)
data=data.drop([3], axis=1)

num_row, num_col = data.shape
selected_column = []

for j in range(num_col-1):
    valcount = data.iloc[:,j].value_counts()
    if '?' not in valcount:
        selected_column.append(j)

np.random.seed(2018)
selected_column = selected_column[:50]
train = np.random.choice([True, False], num_row, replace=True, p=[0.9,0.1])
x_train = data.iloc[train,selected_column].to_numpy()
y_train = data.iloc[train,-1].to_numpy()
x_test = data.iloc[~train,selected_column].to_numpy()
y_test = data.iloc[~train,-1].to_numpy()
x=np.r_[x_train,x_test]
y=np.r_[y_train,y_test]

degrees = [1, 2, 3, 4]
train_means_list = []
means_list = []

for degree in degrees:
    polynomial_features = PolynomialFeatures(degree=degree)
    
    
    polynomial_features.fit(x_train, y_train)  # batch_size

    scores = cross_val_score(
        polynomial_features, x_train, y_train, scoring="neg_mean_squared_error", cv=10
    )
    train_means_list.append(scores.mean())
    scores = cross_val_score(
        polynomial_features, x_test, y_test, scoring="neg_mean_squared_error", cv=10
    )
    means_list.append(scores.mean())

plt.plot(degrees, means_list, label="test")
plt.plot(degrees, train_means_list, label="train")
plt.title("Degree {}".format(degrees))
plt.savefig('test_neg_mse_Polynominal ' + str(degrees) + ' accurance')
plt.clf()