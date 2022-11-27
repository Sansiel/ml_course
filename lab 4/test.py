# In this notebook is demonstrated an proposal for treatment of imbalanced data
# import of needed libraries
import numpy as np
import pandas as pd
from scipy.sparse import *
from sklearn import preprocessing

# evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, f1_score

# classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

data_origin = pd.read_csv("C:\projects\ml_course-main\Lab 4\example_wearablecomputing_weight_lifting_exercises_biceps_curl_variations.csv")

def get_encoded_data(data):
    vars_to_dummy = np.array(data.select_dtypes(include=['object', 'bool']).columns)
    new_data = pd.get_dummies(data, columns=vars_to_dummy, dummy_na=True)
    new_data = new_data.fillna(0)
    return new_data

def evaluate_model(model, model_name, X, y):
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size = 0.25, random_state=1)
    
    # Масштабирование признаков
    X_train_scaled = preprocessing.scale(X_train)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    f_measure = f1_score(y_true, y_pred, average='macro')
    # mat = confusion_matrix(y_true, y_pred)
    print('************************************************************************************************')
    print(model_name, ' accuracy: ', acc)
    print(model_name, ' f-measure: ', f_measure)
    # print(model_name, ' matrix confusion: ')
    # print(mat)
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    print(model_name, ' scores by cv: ', scores)
    print(model_name, ' scores (mean)', np.average(scores))

pd.set_option('display.max_columns', 10)
# encoding categorical variables
y_raw = data_origin['classe']
X_raw = data_origin.drop('classe', axis=1)
X_raw_encoded = get_encoded_data(X_raw)

alphas = [0.1, 0.001]
# learning_rate = 0.0001
for alpha in alphas:
    mlp = MLPClassifier(hidden_layer_sizes=(50, 20, 10), random_state=1, alpha=alpha)
    evaluate_model(mlp, 'MLP', X_raw_encoded, y_raw)
    perc = Perceptron(tol=1e-3, random_state=1, alpha=alpha)
    evaluate_model(perc, 'Perceptron', X_raw_encoded, y_raw)