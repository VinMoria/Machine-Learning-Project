# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:58:40 2024

@author: yyyyyy
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
from datetime import datetime
import os
import json

TRAIN_SET_PATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20

# 定义神经网络模型创建函数
def create_model(learning_rate=0.01, neurons=64):
    model = Sequential()
    model.add(Dense(neurons, input_dim=FEATURE_SELECT_TOP, activation='relu'))  # 隐藏层
    model.add(Dense(1, activation='linear'))  # 输出层，用于回归任务
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 自定义 KerasClassifier 用于 GridSearchCV
class MyKerasClassifier:
    def __init__(self, build_fn=create_model, epochs=10, batch_size=32, learning_rate=0.01, neurons=64, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.neurons = neurons
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y):
        self.model_ = self.build_fn(learning_rate=self.learning_rate, neurons=self.neurons)
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        predictions = self.model_.predict(X)
        return -mean_squared_error(y, predictions)  # 返回负的均方误差，符合 GridSearchCV 的标准

def train_NN_for_sector(Sector):
    # 读取数据集
    df = pd.read_csv(TRAIN_SET_PATH + Sector + ".csv")

    # log 方法
    df["Market Cap(M)"] = np.log(df["Market Cap(M)"].replace(0, np.nan))
    y = df["Market Cap(M)"]

    # 读取选取的特征值
    df_features = pd.read_csv(FEATURE_PATH + Sector + "_feature_importance.csv")
    df_features.columns.values[0] = 'Feature'
    top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])

    X = df[top_features]
    print(X.shape)

    # 归一化
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(list(X.iloc[0]))

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(y_test.describe())

    # 定义参数网格
    param_grid = {
        'epochs': [50, 100],
        'batch_size': [5, 10],
        'learning_rate': [0.01, 0.001],
        'neurons': [32, 64]
    }

    # 使用 GridSearchCV 进行参数搜索
    grid_search = GridSearchCV(
        estimator=MyKerasClassifier(), param_grid=param_grid, scoring="neg_mean_squared_error", cv=5, verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print("Best parameters found: ", grid_search.best_params_)

    # 使用最佳参数训练最终模型
    best_model = grid_search.best_estimator_

    # 在验证集上进行预测
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}")

    # 保存模型
    saved_filename = f"{Sector}_{datetime.now().strftime('%m%d%H%M')}.ml"
    with open(f"NN_model/{saved_filename}", "wb") as f:
        pickle.dump(best_model, f)
    print(f"save file: {saved_filename}")

    res_dict = {
        "Sector": Sector,
        "Features": top_features,
        "RMSE": rmse
    }
    return res_dict

# ==================  main start  ==================
# 清空NN_model下的文件
folder_path = "NN_model"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
print("old models deleted")

sector_list = [
    "Healthcare", "Basic Materials", "Financial", "Consumer Defensive", "Industrials",
    "Technology", "Consumer Cyclical", "Real Estate", "Communication Services", "Energy", "Utilities",
]

res_dict_list = []
for sector in sector_list:
    print(f"=============== {sector} start ===============")
    res_dict_list.append(train_NN_for_sector(sector))

with open(f'NN_model/res.json', 'w') as f:
    json.dump(res_dict_list, f, indent=4)

