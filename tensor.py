import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers,regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import keras_tuner as kt
from kerastuner import HyperParameters, Objective
from kerastuner import Hyperband
from sklearn.metrics import r2_score

# 固定随机种子以确保可复现性
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 数据读取和预处理
FILEPATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20
Sector = 'Utilities'

# 读取主数据集
df = pd.read_csv(os.path.join(FILEPATH, f"{Sector}.csv"))

# 处理“Market Cap(M)”列：替换0为NaN，然后取自然对数
df["Market Cap(M)"] = np.log(df["Market Cap(M)"].replace(0, np.nan))

# 目标变量
y = df["Market Cap(M)"]

# 特征选择
df_features = pd.read_csv(os.path.join(FEATURE_PATH, f"{Sector}_feature_importance.csv"))
df_features.columns.values[0] = 'Feature'
top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])

# 特征矩阵
X = df[top_features].copy()

# 检查特征矩阵中的 NaN 和无穷值
nan_features = X.columns[X.isna().any()].tolist()
inf_features = X.columns[np.isinf(X).any()].tolist()

print("包含 NaN 的特征:", nan_features)
print("包含无穷值的特征:", inf_features)

# 处理包含无穷值的特征
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# 初始化填充器，使用中位数填充 NaN
imputer = SimpleImputer(strategy='median')

# 对特征矩阵进行填充
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 移除目标变量中的极端值
lower_bound = y.quantile(0.01)
upper_bound = y.quantile(0.99)
y = y.clip(lower=lower_bound, upper=upper_bound)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=SEED)

# 特征缩放（标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # 标准化目标变量
# y_scaler = StandardScaler()
# y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
# y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))



def build_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1],)))
    
    # 增加隐藏层和神经元数量
    for i in range(hp.Int('num_layers', 2, 4)):  # 尝试更多层
        units = hp.Int(f'units_{i}', min_value=64, max_value=256, step=32)  # 增加神经元范围
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(hp.Choice('l2_' + str(i), values=[1e-5, 1e-4, 1e-3]))))
        model.add(layers.BatchNormalization())
        
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)  # 调整Dropout范围
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))  # 输出层

    learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-2, sampling='LOG')  # 使用对数取样
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae',rmse])
    return model
# -------------------- 设置 Keras Tuner 并执行网格搜索 --------------------
# 初始化 Keras Tuner 的 GridSearch
tuner = kt.GridSearch(
    build_model,
    objective=Objective("val_loss", direction="min"),
    max_trials=20,  # 增加尝试次数以探索更多超参数组合
    directory='my_dir',
    project_name='grid_search_nn',
    overwrite=True,
    seed=SEED
)

# 执行超参数搜索
tuner.search(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_rmse', patience=10, restore_best_weights=True, mode='min'),
        keras.callbacks.TerminateOnNaN()
    ]
)

# 获取最佳超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("最佳超参数组合：")
print(f"隐藏层数量: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"第 {i+1} 层的神经元数量: {best_hps.get(f'units_{i}')}")
    print(f"第 {i+1} 层的 L2 正则化: {best_hps.get(f'l2_{i}')}")
    print(f"第 {i+1} 层的 Dropout 率: {best_hps.get(f'dropout_{i}')}")
print(f"学习率: {best_hps.get('learning_rate')}")

# -------------------- 构建并训练最佳模型 --------------------
# 构建最佳模型
model = tuner.hypermodel.build(best_hps)

# 训练最佳模型
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_rmse', patience=10, restore_best_weights=True, mode='min'),
        keras.callbacks.TerminateOnNaN()
    ]
)

# -------------------- 评估最佳模型 --------------------
# 在测试集上评估模型
test_loss, test_mae, test_rmse = model.evaluate(X_test_scaled, y_test, verbose=2)

# 使用模型进行预测
#y_pred_scaled = model.predict(X_test_scaled)

# # 反标准化预测值和实际值
# y_test_original = y_scaler.inverse_transform(y_test_scaled)
# y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

# # 在测试集上评估模型
# test_loss, test_mae, test_rmse = model.evaluate(X_test_scaled, y_test_scaled, verbose=2)

# 手动计算 MSE 和 RMSE
#mse_manual = mean_squared_error(y_test_scaled, y_pred_scaled)
#rmse_manual = mse_manual ** 0.5

print('模型评估的均方误差 (MSE):', test_loss)
#print('手动计算的均方误差 (MSE):', mse_manual)
print('模型评估的均方根误差 (RMSE):', test_rmse)
#print('手动计算的均方根误差 (RMSE):', rmse_manual)