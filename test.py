#!/usr/bin/env python
# coding: utf-8
from tic_test import load_tic_test
from sklearn.naive_bayes import BernoulliNB
import joblib
import numpy as np

print("加载 TIC 2000 测试数据\n")
dataset = load_tic_test()

X = dataset.data
y = dataset.target
headers = dataset.feature_names

print("数据集")
print("已加载 %d 测试数据集（离散数据）, 包含 %d 个特征" % X.shape)
print("预测最有可能购买房车险的800名客户，寻找与实际符合最多的算法，实际存在 %d 个\n" % y.sum())

train_features = [86, 87, 43, 44, 47, 59, 65, 68, 80]

print("过滤特征")
print("\n".join(["%-2d %s" % (idx, headers[idx - 1]) for idx in train_features]))
X = X[:, [i - 1 for i in train_features]]

print("\n模型测试")
clf = joblib.load('tic-BernoulliNB.pkl')
pred_y = clf.predict(X)
probs = clf.predict_proba(X)

pred_y_y = np.hstack((pred_y.reshape(pred_y.shape[0], 1), y.reshape(y.shape[0], 1)))

probs = np.hstack((probs, pred_y_y))

column_names = ['prob_neg', 'prob_pos', 'pred_y', 'y']
probs.dtype = [(n, probs.dtype) for n in column_names]

probs = np.sort(probs, axis=0, order=['pred_y'])

print("\n测试结果")
best800 = probs[probs.shape[0] - 800:probs.shape[0], :]['y'].sum()

print("最可能的800个客户中，准确命中了 %d 个客户。" % best800)
