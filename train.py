#!/usr/bin/env python
# coding: utf-8
from tic import load_tic
from sklearn.naive_bayes import BernoulliNB
import joblib

print("加载 TIC 2000 数据\n")
dataset = load_tic((86, 0, 19, 20))

X = dataset.data
y = dataset.target
headers = dataset.feature_names

print("数据集")
print("已加载 %d 训练数据集（离散数据）, 包含 %d 个特征" % X.shape)
print("预测目标购买房车险的客户共 %d 个\n" % y.sum())

BernoulliNB_alpha = 0.01
train_features = [86, 87, 43, 44, 47, 59, 65, 68, 80]

print("选择训练特征")
print("\n".join(["%-2d %s" % (idx, headers[idx - 1]) for idx in train_features]))
X = X[:, [i - 1 for i in train_features]]

print("\n训练模型")
clf = BernoulliNB(alpha=BernoulliNB_alpha)
model = clf.fit(X, y)

joblib.dump(clf, 'tic-BernoulliNB.pkl')

# 训练结果
print("\n评分")
print(model.score(X, y))
