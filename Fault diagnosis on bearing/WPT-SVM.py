from scipy.io import loadmat
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np

m1 = loadmat("all.mat")
x1 = m1['all_DE_time']

a1 = 0
for i in range(500):
    a1 = sum(x1[i])
    for j in range(8):
        x1[i][j] = x1[i][j]/a1

train_data = x1
y1 = torch.zeros(25).int()
y2 = torch.ones(25).int()
y3 = y2+1
y4 = y3+1
y5 = y4+1
y6 = y5+1
y7 = y6+1
y8 = y7+1
y9 = y8+1
y10 = y9+1
train_target = torch.cat([y1, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y2, y3, y4, y5, y6, y7, y8, y9, y10], 0)
train_target = train_target.numpy()

x_train, x_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.3)

clf = SVC(C=10, kernel='rbf', gamma=1, decision_function_shape='rbf')
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))  # 精度
pred_y = clf.predict(x_test)
accuracy = sum(pred_y == y_test)/len(x_test)
print(accuracy)

# knn = KNeighborsClassifier(n_neighbors=4)
# knn.fit(x_train, y_train)
# pred_y = knn.predict(x_test)
# print(pred_y)
# accuracy = sum(pred_y == y_test)/len(x_test)
# print(accuracy)
