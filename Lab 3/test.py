import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:\projects\ml_course-main\Lab 3\communities.data', header=None)
data = data.drop([3], axis=1)

num_row, num_col = data.shape
selected_column = [43, 44, 45, 46, 47, 54, 55, 124]
# -- MalePctDivorce: percentage of males who are divorced (numeric - decimal)
# -- MalePctNevMarr: percentage of males who have never married (numeric - decimal)
# -- FemalePctDiv: percentage of females who are divorced (numeric - decimal)
# -- TotalPctDiv: percentage of population who are divorced (numeric - decimal)
# -- PersPerFam: mean number of people per family (numeric - decimal)
# -- NumIlleg: number of kids born to never married (numeric - decimal)
# -- PctIlleg: percentage of kids born to never married (numeric - decimal)
# -- ViolentCrimesPerPop: total number of violent crimes per 100K popuation (numeric - decimal) GOAL attribute (to be predicted)

np.random.seed(2018)
train = np.random.choice([True, False], num_row, replace=True, p=[0.9, 0.1])
x_train = data.iloc[train, selected_column].to_numpy()
y_train = data.iloc[train, -1].to_numpy()
x_test = data.iloc[~train, selected_column].to_numpy()
y_test = data.iloc[~train, -1].to_numpy()
x = np.r_[x_train, x_test]
y = np.r_[y_train, y_test]

# build linear model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_predict = regr.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_predict)

print('\t', 'linear regression model:')
print('Mean squared error on test set:', mse)
print("Coefficient of determination: %.2f" % metrics.r2_score(y_test, y_predict))

# X_scaled: float = preprocessing.scale(x)
# x_train, X_test, y_train, y_true = train_test_split(x, y, test_size=0.25, random_state=1)

# Polynominal

degrees = [1, 2, 3] #, 4, 5, 6, 7, 8] #, 9, 10, 11, 12, 13, 14, 15]
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

plt.plot(degrees, accuracy_score_list, c='blue', label="test")
plt.plot(degrees, train_accuracy_score_list, c='red', label="train")
plt.title("Degree {}".format(degrees))
plt.xlabel("degrees")
plt.ylabel("accuracy")
plt.savefig('score_Polynominal ' + str(degrees) + ' accurance')
plt.clf()

# Lasso

n_alphas = 200
alpha_start = 0.001
alpha_end = 0.1
alphas = np.linspace(alpha_start, alpha_end, n_alphas)
model = linear_model.Lasso()
train_accuracy_score_list_for_alpha = []
accuracy_score_list_for_alpha = []

for alpha in alphas:
    model.set_params(alpha=alpha)
    model.fit(x_train, y_train)
    accuracy_score_list_for_alpha.append(model.score(x_test, y_test))
    train_accuracy_score_list_for_alpha.append(model.score(x_train, y_train))

plt.plot(alphas, accuracy_score_list_for_alpha, c='blue', label="test")
plt.plot(alphas, train_accuracy_score_list_for_alpha, c='red', label="train")
plt.title('Alphas от ' + str(alpha_start) + ' до ' + str(alpha_end) + '  по 200')
plt.xlabel("alphas")
plt.ylabel("accuracy")
plt.savefig('score_Lasso with alpha от ' + str(alpha_start) + ' до ' + str(alpha_end) + ' по 200 accurance.png')
plt.clf()

# degrees = [1, 2, 3] #, 4, 5, 6, 7, 8] #, 9, 10, 11, 12, 13, 14, 15]
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

print(means_list)
print(means_list)
print(train_means_list)
plt.plot(degrees, means_list, c='blue', label="test")
plt.plot(degrees, train_means_list, c='red', label="train")
plt.title("Degree {}".format(degrees))
plt.xlabel("degrees")
plt.ylabel("cross_val_score")
plt.savefig('neg_mse_Polynominal ' + str(degrees) + ' accurance')
plt.clf()

# Lasso
# n_alphas = 200
# alphas = np.linspace(0.1, 10, n_alphas)
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

print(means_list)
print(train_means_list)
plt.plot(alphas, means_list, c='blue', label="test")
plt.plot(alphas, train_means_list, c='red', label="train")
plt.xlabel("alphas")
plt.ylabel("cross_val_score")
plt.title('Alphas от ' + str(alpha_start) + ' до ' + str(alpha_end) + '  по 200')
plt.savefig('neg_mse_Lasso with alpha от ' + str(alpha_start) + ' до ' + str(alpha_end) + ' по 200 accurance.png')
plt.clf()
