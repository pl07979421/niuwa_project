from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data, target = load_iris(return_X_y=True)

# 归一化
scaler = StandardScaler()
x = scaler.fit_transform(data)

train_X, test_X, train_y, test_y = train_test_split(x, target, test_size=0.2)

model = SVC()
model.fit(train_X, train_y)

expected = test_y
predicted = model.predict(test_X)
# 输出指标
print(metrics.classification_report(expected, predicted))
