import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, SpectralClustering
from sklearn.feature_extraction.text import CountVectorizer

# Чтение данных
data_origin = pd.read_csv("SMSSpamCollection.csv")
type_raw = data_origin['type']
text_raw = data_origin['text']

n_clusters = 2
X_scale_bool = True

# Векторизация
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(text_raw)
X_scale = preprocessing.scale(X, with_mean=False)

if X_scale_bool:
    X = X_scale

#  Забавные тесты
# from sklearn.model_selection import train_test_split
# x_train, X_test, y_train, y_test = train_test_split(X, type_raw, test_size=0.25, random_state=1)
# k_means.predict(X_test[1])
# k_means.predict(X_test[70])


# Конструирование кластера KMeans
k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
k_means.fit(X)

# Конструирование кластера DBSCAN
db_scan = DBSCAN(eps=0.3, min_samples=n_clusters)
db_scan.fit(X)

# Конструирование кластера MiniBatchKMeans
mbk = MiniBatchKMeans(
    init="k-means++",
    n_clusters=n_clusters,
    batch_size=45,
    n_init=10,
    max_no_improvement=10,
    verbose=0,
)
mbk.fit(X)

# Конструирование кластера SpectralClustering
spectr = SpectralClustering(n_clusters=n_clusters, random_state=1,  affinity='nearest_neighbors')
x_array = X.toarray()
spectr.fit(x_array)


# k_means_cluster_centers = k_means.cluster_centers_
# mbk_cluster_centers = mbk.cluster_centers_


# Оценка
def score_task(km):
    scores = dict()
    # Описывает близость алгоритма кластеризации к этому совершенству (> -> лучше)
    scores["Homogeneity"] = metrics.homogeneity_score(text_raw, km.labels_)
    # Описывает близость алгоритма кластеризации к этому совершенству (> -> лучше)
    scores["Completeness"] = metrics.completeness_score(text_raw, km.labels_)
    # Среднее между однородностью и полнотой
    scores["V-measure"] = metrics.v_measure_score(text_raw, km.labels_)
    # Выражает схожесть двух разных кластеризаций одной и той же выборки
    # Отрицательные значения соответствуют "независимым" разбиениям на кластеры,
    # значения, близкие к нулю, — случайным разбиениям,
    # положительные значения говорят о том, что два разбиения схожи
    scores["Adjusted Rand-Index"] = metrics.adjusted_rand_score(text_raw, km.labels_)
    # Позволяет оценить качество кластеризации, используя только саму выборку и результат кластеризации
    # Показывает насколько среднее расстояние до объектов своего кластера отличается от среднего расстояния до объектов других кластеров
    # Значения, близкие к -1, соответствуют плохим (разрозненным) кластеризациям,
    # значения, близкие к нулю, говорят о том, что кластеры пересекаются и накладываются друг на друга,
    # значения, близкие к 1, соответствуют "плотным" четко выделенным кластерам.
    scores["Silhouette Coefficient"] = metrics.silhouette_score(X, km.labels_, sample_size=2000)
    return scores


score_kmean = score_task(k_means)
score_mbk = score_task(mbk)
score_db_scan = score_task(db_scan)
score_spectr = score_task(spectr)
print('KMeans ', score_kmean)
print('MiniBatchKMeans ', score_mbk)
print('DBSCAN ', score_db_scan)
print('SpectralClustering ', score_spectr)


import pandas as pd
import matplotlib.pyplot as plt

listHomogeneity = [score_kmean['Homogeneity'], score_mbk['Homogeneity'], score_db_scan['Homogeneity'], score_spectr['Homogeneity']]
listCompleteness = [score_kmean['Completeness'], score_mbk['Completeness'], score_db_scan['Completeness'], score_spectr['Completeness']]
listV = [score_kmean['V-measure'], score_mbk['V-measure'], score_db_scan['V-measure'], score_spectr['V-measure']]
listAdjusted = [score_kmean['Adjusted Rand-Index'], score_mbk['Adjusted Rand-Index'], score_db_scan['Adjusted Rand-Index'], score_spectr['Adjusted Rand-Index']]
listSilhouette = [score_kmean['Silhouette Coefficient'], score_mbk['Silhouette Coefficient'], score_db_scan['Silhouette Coefficient'], score_spectr['Silhouette Coefficient']]

df = pd.DataFrame({'Homogeneity': listHomogeneity,
                   'Completeness': listCompleteness,
                   'V-measure': listV,
                   'Adjusted Rand-Index': listAdjusted,
                   'Silhouette Coefficient': listSilhouette
                   }, index=['KMeans', 'MiniBatchKMeans', 'DBSCAN', 'SpectralClustering'])

ax = df.plot.barh()
if X_scale_bool:
    n_clusters = str(n_clusters) + ' and Scaling'
plt.savefig('clustering methods n_clusters=' + str(n_clusters) + '.png')
