# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:58:40 2024

@author: yyyyyy
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer  # 导入 KNNImputer
import os
from datetime import datetime
import json

TRAIN_SET_PATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20

# 定义创建神经网络模型的函数，增加hidden_layers和dropout
def create_model(learning_rate=0.01, neurons=64, hidden_layers=1, dropout_rate=0.0):
    inputs = Input(shape=(FEATURE_SELECT_TOP,))
    x = Dense(neurons, activation='relu')(inputs)
    
    # 添加额外的隐藏层和Dropout层
    for _ in range(hidden_layers - 1):
        x = Dense(neurons, activation='relu')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    outputs = Dense(1, activation='linear')(x)  # 输出层，用于回归任务
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

class MyKerasClassifier:
    def __init__(self, build_fn=create_model, epochs=10, batch_size=32, learning_rate=0.01, neurons=64, hidden_layers=1, dropout_rate=0.0, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.neurons = neurons
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.model_ = None

    def fit(self, X, y):
        self.model_ = self.build_fn(learning_rate=self.learning_rate, neurons=self.neurons, hidden_layers=self.hidden_layers, dropout_rate=self.dropout_rate)
        
        # 使用 EarlyStopping 回调来防止过拟合
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, validation_split=0.2, callbacks=[early_stopping])
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y):
        predictions = self.model_.predict(X)
        return -mean_squared_error(y, predictions) 

    def get_params(self, deep=True):
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'neurons': self.neurons,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

def train_NN_for_sector(Sector):
    # 读取数据集
    df = pd.read_csv(TRAIN_SET_PATH + Sector + ".csv")

    # log 方法处理 Market Cap(M)
    df["Market Cap(M)"] = np.log(df["Market Cap(M)"].replace(0, np.nan))
    y = df["Market Cap(M)"]

    # 读取选取的特征值
    df_features = pd.read_csv(FEATURE_PATH + Sector + "_feature_importance.csv")
    df_features.columns.values[0] = 'Feature'
    top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])

    X = df[top_features]
    print(X.shape)

    # 使用 KNNImputer 处理缺失值
    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(X)

    # 归一化
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=top_features)
    print(list(X.iloc[0]))

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(y_test.describe())

    # 定义参数网格，增加hidden_layers和dropout_rate
    param_grid = {
    'epochs': [50, 100],
    'batch_size': [5, 10],
    'learning_rate': [0.01, 0.001],
    'neurons': [64, 128],  # 增加神经元数量
    'hidden_layers': [2, 3],  # 增加隐藏层数量
    'dropout_rate': [0.0, 0.2]
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
    rmse = mse ** 0.5
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}")

    # 保存模型
    saved_filename = f"{Sector}_{datetime.now().strftime('%m%d%H%M')}.h5"
    if not os.path.exists("NN_model"):
        os.makedirs("NN_model")
    best_model.model_.save(f"NN_model/{saved_filename}")
    print(f"Model saved as: {saved_filename}")

    res_dict = {
        "Sector": Sector,
        "Features": top_features,
        "RMSE": rmse
    }
    return res_dict

train_NN_for_sector("Utilities")



