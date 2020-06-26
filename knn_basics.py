import cv2
import numpy as np
import matplotlib.pyplot as plt

train_data = np.random.randint(0, 100, (25, 2)).astype(np.float32)
labels = np.random.randint(0, 2, (25, 1))

class_a = train_data[labels.ravel() == 1]
class_b = train_data[labels.ravel() == 0]

plt.scatter(class_a[:, 0], class_a[:, 1], color='r', marker='^')
plt.scatter(class_b[:, 0], class_b[:, 1], color='b', marker='o')
plt.show()

test_data = np.random.randint(0, 100, (1, 2)).astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)
ret, results, neighbours, dist = knn.findNearest(test_data, 3)

plt.scatter(class_a[:, 0], class_a[:, 1], color='r', marker='^')
plt.scatter(class_b[:, 0], class_b[:, 1], color='b', marker='o')
plt.scatter(test_data[:, 0], test_data[:, 1], s=180, color=('r' if ret == 1 else 'g'), marker='X')

plt.show()
