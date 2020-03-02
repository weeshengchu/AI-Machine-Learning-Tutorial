import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
# scale down all the digits dataset to -1 and 1 for training
data = scale(digits.data)
y = digits.target
# print(y)

# set the amount of cluster k
# k = len(np.unique(y))
k = 10
samples, features = data.shape

# https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
# higher the better, higher level maths
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# init -> depends on running time, n_init -> iterations
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

# 1        	69405	0.603	0.652	0.627	0.466	0.623	0.147
