
import numpy as np
from numpy import mean
from numpy import std
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.svm import SVC
from numpy.lib.shape_base import dstack
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


dataset = pd.read_csv('Dataset.txt', header=0)
attack_types = pd.read_csv('Attack_types.txt', header = 0)

x = dataset.loc[ : , dataset.columns != 'attack_category']
y=dataset.loc[ : , dataset.columns == 'attack_category']
z=attack_types.loc[ : , ]

df1 = pd.DataFrame(data=y)
df2 = pd.DataFrame(data=attack_types)

for i in df2.index:
    df1 = df1.replace([df2.iloc[i].attack_category], df2.iloc[i].attack_type)


LE = LabelEncoder()
df1['attack_category'] = LE.fit_transform(df1['attack_category'])

x['protocol_type']= LE.fit_transform(x['protocol_type'])
x['service']= LE.fit_transform(x['service'])
x['flag']= LE.fit_transform(x['flag'])



kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(x)


y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
label_names = kmeans.labels_

# x_train, X_test, y_train, y_test = train_test_split(x, df1, test_size = 0.2, random_state = 0)


# label_names= ["dos", "u2r" , "r2l", "probe","normal"]

# seed = np.random.seed(50)
# tf.random.set_seed(50)
# kfold = model_selection.KFold(n_splits=10, random_state= seed)
# models = []
# estimators =  []

# model1 = keras.models.Sequential()
# model1.add(keras.layers.Flatten(input_shape=[42]))
# model1.add(keras.layers.Dense(64 , activation="sigmoid"))
# model1.add(keras.layers.Dense(5, activation="softmax"))
# model1.summary()
# model1.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
# model_history = model1.fit(x_train,y_train,epochs=5) 

# model2 = keras.models.Sequential()
# model2.add(keras.layers.Flatten(input_shape=[42]))
# model2.add(keras.layers.Dense(64 , activation="sigmoid"))
# model2.add(keras.layers.Dense(32 , activation="sigmoid"))
# model2.add(keras.layers.Dense(5, activation="softmax"))
# model2.summary()
# model2.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
# model_history = model2.fit(x_train,y_train,epochs=5) 

# model3 = keras.models.Sequential()
# model3.add(keras.layers.Flatten(input_shape=[42]))
# model3.add(keras.layers.Dense(32 , activation="sigmoid"))
# model3.add(keras.layers.Dense(5, activation="softmax"))
# model3.summary()
# model3.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
# model_history = model1.fit(x_train,y_train,epochs=5) 




# models = [model1, model2, model3]




# #evaluate the models and store results
# results = None
# for  model in models:
#     scores = model.predict(X_test)
    
#     if results is None:
#         results = scores
#         continue
#     results = dstack((results,scores))

# results = results.reshape(
#     (results.shape[0], results.shape[1] * results.shape[2]))
 


# final_model = LogisticRegression(max_iter=1000)
# final_model.fit(results, y_test)
# predictions = final_model.predict(results)

# print("Final_Model_Score:")
# print(accuracy_score(y_test, predictions))



