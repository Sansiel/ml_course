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

data = pd.read_csv('C:\projects\ml_course-main\Lab 3\communities.data', header=None)
data=data.drop([3], axis=1)
num_row, num_col = data.shape
selected_column = []

for j in range(num_col-1):
    valcount = data.iloc[:,j].value_counts()
    if '?' not in valcount:
        selected_column.append(j)
    elif valcount['?'] < 0.01 * num_row:
        valmean = pd.to_numeric(data.iloc[:,j], errors='coerce').mean()
        for i in range(num_row):
            if data.iloc[i,j] == '?':
                data.iloc[i,j] = valmean
        data.iloc[:,j] = pd.to_numeric(data.iloc[:,j])
        selected_column.append(j)

np.random.seed(2018)
train = np.random.choice([True, False], num_row, replace=True, p=[0.9,0.1])
x_train = data.iloc[train,selected_column].to_numpy()
y_train = data.iloc[train,-1].to_numpy()
x_test = data.iloc[~train,selected_column].to_numpy()
y_test = data.iloc[~train,-1].to_numpy()
x=np.r_[x_train,x_test]
y=np.r_[y_train,y_test]

def sort_colum(multi_array):
    array = []
    for arr in multi_array:
        array.append(arr[6])
    array.sort()
    return array

#build linear model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_predict=regr.predict(x_test)
mse=metrics.mean_squared_error(y_test, y_predict)

print('\t','linear regression model:')
print('Mean squared error on test set:', mse)
print("Coefficient of determination: %.2f" % metrics.r2_score(y_test, y_predict))

plt.scatter(sort_colum(x_test), y_test, color="black")
plt.plot(sort_colum(x_test), y_predict, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())
plt.title( "Linear MSE = {})".format(mse ) )
plt.savefig('linear')
plt.clf()


# Polynominal
# x_train = x_train[:60]
# y_train = y_train[:60]
# x_test = x_test[:60]
# y_test = y_test[:60]
degrees = [1, 2]#, 3, 4]
for degree in degrees:
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)

    linear_regression = linear_model.LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    pipeline.fit(x_train, y_train)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(
        pipeline, x_train, y_train, scoring="neg_mean_squared_error", cv=10
    )
    print('\t','polynominal regression model:')
    print('Cross value score:', scores)
    # Cross value score: [ -77.33569045  -57.2715422   -28.44285826 -227.31448579   -0.446458
    #  -182.14188518   -1.72085151  -17.56463659  -32.19301248  -62.89467899]

    y_pred_poly = pipeline.predict(x_test)

    plt.plot(sort_colum(x_test), y_pred_poly, label="Model")
    plt.scatter(sort_colum(x_test), y_test, edgecolor="b", s=20, label="Samples")
    plt.title(
            "Degree {}\nMSE = {:.2e}(+/- {:.2e})\nMetrics.MSE = {}".format(
                degree, -scores.mean(), scores.std(), metrics.mean_squared_error(y_test, y_pred_poly)
            )
        )
    plt.savefig('Polynominal ' + str(degree) + ' not optimized')
    plt.clf()

# Lasso
alpha_list = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
for alpha in alpha_list:
    reg = linear_model.Lasso(alpha=alpha)
    reg.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    plt.plot(sort_colum(x_test), y_pred, label="Model")
    plt.scatter(sort_colum(x_test), y_test, edgecolor="b", s=20, label="Samples")
    plt.title(metrics.mean_squared_error(y_test, y_pred))
    plt.savefig('Lasso with alpha - ' + str(alpha) + '.png')
    plt.clf()

# Interesting

def my_linspace(min_value, max_value, steps):
    diff = max_value - min_value
    return np.linspace(min_value - 0.1 * diff, max_value + 0.1 * diff, steps)

steps = 200
x0 = my_linspace(min(x[:,66]), max(x[:,66]), steps)
x1 = my_linspace(min(x[:,43]), max(x[:,43]), steps)
xx0, xx1 = np.meshgrid(x0, x1)
mesh_data = np.c_[xx0.ravel(), xx1.ravel()]
regr1 = linear_model.LinearRegression()
regr1.fit(x_train[:,[66,43]], y_train)
mesh_y = regr1.predict(mesh_data).reshape(steps,steps)
plt.contourf(xx0, xx1, mesh_y, 20, cmap=plt.cm.Greys, alpha=0.5)
plt.savefig('interesting')