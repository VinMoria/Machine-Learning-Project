# 导入必要的库
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
!pip install tensorflow scikit-learn
!pip install scikit-learn

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# 加载数据
iris = load_iris()
X = iris.data
Y = iris.target

# 编码标签
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)

# 标准化数据（对神经网络的训练有帮助）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

# 定义模型创建函数
def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(8, input_dim=4, kernel_initializer=init, activation='relu'))  # 输入层和第一隐藏层
    model.add(Dense(4, kernel_initializer=init, activation='relu'))  # 第二隐藏层
    model.add(Dense(3, kernel_initializer=init, activation='softmax'))  # 输出层
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 包装Keras模型，使其可以与sklearn的GridSearchCV一起使用
model = KerasClassifier(build_fn=create_model, verbose=0)

# 定义超参数网格
param_grid = {
    'batch_size': [10, 20, 40],         # 批大小
    'epochs': [10, 50],                 # 训练轮次
    'optimizer': ['SGD', 'Adam'],       # 优化器
    'init': ['glorot_uniform', 'normal']  # 权重初始化方法
}

# 使用GridSearchCV进行超参数搜索
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# 训练模型并搜索最佳参数组合
grid_result = grid.fit(X_train, Y_train)

# 输出最佳参数和最佳得分
print(f"Best Accuracy: {grid_result.best_score_:.4f} using {grid_result.best_params_}")

# 输出每个参数组合的评估结果 
means = grid_result.cv_results_['mean_test_score'] stds = grid_result.cv_results_['std_test_score'] params = grid_result.cv_results_['params'] for mean, std, param in zip(means, stds, params): 
    print(f"Mean Accuracy: {mean:.4f} (std: {std:.4f}) with: {param}") 

# 在测试集上评估模型的性能 
best_model = grid_result.best_estimator_ 
test_accuracy = best_model.score(X_test, Y_test) print(f"Test Accuracy: {test_accuracy:.4f}")
