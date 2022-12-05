# import os
# print(os.listdir(".."))

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
    # elif valcount['?'] < 0.01 * num_row:
    #     valmean = pd.to_numeric(data.iloc[:,j], errors='coerce').mean()
    #     for i in range(num_row):
    #         if data.iloc[i,j] == '?':
    #             data.iloc[i,j] = valmean
    #     data.iloc[:,j] = pd.to_numeric(data.iloc[:,j])
    #     selected_column.append(j)

np.random.seed(2018)
selected_column = selected_column[:50]
train = np.random.choice([True, False], num_row, replace=True, p=[0.9,0.1])
x_train = data.iloc[train,selected_column].to_numpy()
y_train = data.iloc[train,-1].to_numpy()
x_test = data.iloc[~train,selected_column].to_numpy()
y_test = data.iloc[~train,-1].to_numpy()
x=np.r_[x_train,x_test]
y=np.r_[y_train,y_test]


# def sort_colum(multi_array):
#     array = []
#     for arr in multi_array:
#         array.append(arr[6])
#     array.sort()
#     return array

#build linear model
# regr = linear_model.LinearRegression()
# regr.fit(x_train, y_train)
# y_predict=regr.predict(x_test)
# mse=metrics.mean_squared_error(y_test, y_predict)

# print('\t','linear regression model:')
# print('Mean squared error on test set:', mse)
# print("Coefficient of determination: %.2f" % metrics.r2_score(y_test, y_predict))

# plt.scatter(sort_colum(x_test), y_test, color="black")
# plt.plot(sort_colum(x_test), y_predict, color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())
# plt.title( "Linear MSE = {})".format(mse ) )
# plt.savefig('linear')
# plt.clf()

# x = data.iloc[selected_column[:-1]].to_numpy()
# y = data.iloc[selected_column[-1:]].to_numpy()

X_scaled: float = preprocessing.scale(x)
x_train, X_test, y_train, y_true = train_test_split(x, y, test_size = 0.25, random_state=1)


# Polynominal
# x_train = x_train[:60]
# y_train = y_train[:60]
# x_test = x_test[:60]
# y_test = y_test[:60]
degrees = [8]
train_accuracy_score_list = []
accuracy_score_list = []

for degree in degrees:
    polynomial_features = PolynomialFeatures(degree=degree)

    linear_regression = linear_model.LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    pipeline.fit(x_train, y_train)  # batch_size
    accuracy_score_list.append(pipeline.score(x_test, y_test))
    train_accuracy_score_list.append(pipeline.score(x_train, y_train))

    # scores = cross_val_score(
    #     pipeline, x_train, y_train, scoring="neg_mean_squared_error", cv=10
    # )
    # print('\t','polynominal regression model:')
    # print('Cross value score:', scores)
    # means_list.append(scores.mean())

plt.plot(degrees, accuracy_score_list, label="test")
plt.plot(degrees, train_accuracy_score_list, label="train")
plt.title("Degree {}".format(degrees))
plt.savefig('score_Polynominal ' + str(degrees) + ' accurance')
plt.clf()
print(accuracy_score_list)

# Lasso

n_alphas = 200
alphas = np.linspace(0.1, 10, n_alphas)
model = linear_model.Lasso()
train_accuracy_score_list_for_alpha = []
accuracy_score_list_for_alpha = []

for alpha in alphas:
    model.set_params(alpha=alpha)
    model.fit(x_train, y_train)
    accuracy_score_list_for_alpha.append(model.score(x_test, y_test))
    train_accuracy_score_list_for_alpha.append(model.score(x_train, y_train))
    
plt.plot(alphas, accuracy_score_list_for_alpha, label="test")
plt.plot(alphas, train_accuracy_score_list_for_alpha, label="train")
plt.title("Alphas от 0.1 до 10 по 200")
plt.savefig('score_Lasso with alpha от 0.1 до 10 по 200 accurance.png')
print(accuracy_score_list_for_alpha)
plt.clf()


degrees = [1, 2, 3, 4]
train_means_list = []
means_list = []

for degree in degrees:
    polynomial_features = PolynomialFeatures(degree=degree)

    linear_regression = linear_model.LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    pipeline.fit(x_train, y_train)

    scores = cross_val_score(
        pipeline, x_train, y_train, scoring="neg_mean_squared_error", cv=10
    )
    train_means_list.append(scores.mean())
    scores = cross_val_score(
        pipeline, x_test, y_test, scoring="neg_mean_squared_error", cv=10
    )
    means_list.append(scores.mean())

plt.plot(degrees, means_list, label="test")
plt.plot(degrees, train_means_list, label="train")
plt.title("Degree {}".format(degrees))
plt.savefig('neg_mse_Polynominal ' + str(degrees) + ' accurance')
plt.clf()

# Lasso
n_alphas = 200
alphas = np.linspace(0.1, 10, n_alphas)
model = linear_model.Lasso()
train_means_list = []
means_list = []

for alpha in alphas:
    model.set_params(alpha=alpha)
    model.fit(x_train, y_train)
    scores = cross_val_score(
        model, x_train, y_train, scoring="neg_mean_squared_error", cv=10
    )
    train_means_list.append(scores.mean())
    scores = cross_val_score(
        model, x_test, y_test, scoring="neg_mean_squared_error", cv=10
    )
    means_list.append(scores.mean())
    
plt.plot(alphas, means_list, label="test")
plt.plot(alphas, train_means_list, label="train")
plt.title("Alphas от 0.1 до 10 по 200")
plt.savefig('neg_mse_Lasso with alpha от 0.1 до 10 по 200 accurance.png')
plt.clf()
