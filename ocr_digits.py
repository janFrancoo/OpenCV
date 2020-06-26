import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv")
labels = data.pop("label")

digits = []
for index, row in data.iterrows():
    digits.append(np.float32(row.to_numpy()))

x_train, x_test, y_train, y_test = train_test_split(digits, labels, test_size=0.2)

x_train = np.array(x_train).astype(np.float32)
x_test = np.array(x_test).astype(np.float32)
y_train = np.array(y_train)
y_test = np.array(y_test)

knn = cv2.ml.KNearest_create()
knn.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
ret, results, neighbours, dist = knn.findNearest(x_test, k=7)

matches = results == x_test
correct = np.count_nonzero(matches)
accuracy = correct / results.size
print(accuracy)
