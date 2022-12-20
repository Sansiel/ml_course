import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, f1_score

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

data_origin = pd.read_csv("example_wearablecomputing_weight_lifting_exercises_biceps_curl_variations.csv")


def get_encoded_data(data):
    vars_to_dummy = np.array(data.select_dtypes(include=['object', 'bool']).columns)
    new_data = pd.get_dummies(data, columns=vars_to_dummy, dummy_na=True)
    new_data = new_data.fillna(0)
    return new_data


def evaluate_model(model, model_name, X, y, alpha):
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.25, random_state=1)

    # Масштабирование признаков
    X_train_scaled = preprocessing.scale(X_train)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f_measure = f1_score(y_true, y_pred, average='macro')
    print('************************************************************************************************')
    print('Проеверка модели ', model_name, ' на alpha = ', alpha)
    print(model_name, ' accuracy: ', acc)
    print(model_name, ' f-measure: ', f_measure)
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)  # Крос валидатор 5 раз с рандомизацией каждые 2
    # cv = KFold(n_splits=5)
    # cv = LeaveOneOut()

    print("Точность модели на полной выборке")
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')  # Эт чем ближе к 1, тем лучше
    print(model_name, ' scores by cv: ', scores)
    print(model_name, ' scores (mean)', np.average(scores))

    print("Точность модели на обучающей выборке")
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
    print(model_name, ' scores by cv: ', scores)
    print(model_name, ' scores (mean)', np.average(scores))

    print("Точность модели на тестовой выборке")
    scores = cross_val_score(model, X_test, y_true, cv=cv, scoring='f1_macro')
    print(model_name, ' scores by cv: ', scores)
    avg_score = np.average(scores)
    print(model_name, ' scores (mean)', avg_score)
    return avg_score


pd.set_option('display.max_columns', 10)
y_raw = data_origin['classe']
X_raw = data_origin.drop('classe', axis=1)
X_raw_encoded = get_encoded_data(X_raw)

alphas = [0.1, 0.01, 0.001, 0.0001]
avg_scores_mlp = []
avg_scores_perceptron = []
# learning_rate = 0.001  # коэффициент обучения
# tol = 1e-3  # оптимизация
# hidden_layer_sizes - это позволяет установить количество слоев и количество узлов в классификаторе нейронной сети
for alpha in alphas:
    mlp = MLPClassifier(hidden_layer_sizes=(50, 20, 10), random_state=1, alpha=alpha)  # 3 слоя
    avg_scores_mlp.append(evaluate_model(mlp, 'MLP', X_raw_encoded, y_raw, alpha))
    perc = Perceptron(tol=1e-3, random_state=1, alpha=alpha)
    avg_scores_perceptron.append(evaluate_model(perc, 'Perceptron', X_raw_encoded, y_raw, alpha))  # Критерий остановки

plt.plot(alphas, avg_scores_mlp, c='blue', label="MLP")
plt.plot(alphas, avg_scores_perceptron, c='red', label="Perceptron")
plt.title("MLP vs Perceptron")
plt.savefig('MLP vs Perceptron.png')