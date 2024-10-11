import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import keras_tuner as kt
from kerastuner import HyperParameters, Objective

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

# 标准化目标变量
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# 构建模型函数
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1],)))
    
    # 增加隐藏层和神经元数量
    for i in range(hp.Int('num_layers', 2, 3)):  # 降低层数上限
        units = hp.Int(f'units_{i}', min_value=64, max_value=192, step=32)  # 缩小神经元范围
        
        # 选择激活函数
        activation = hp.Choice('activation', values=['relu', 'swish'])  # 移除 tanh
        model.add(layers.Dense(units, activation=activation, 
                               kernel_regularizer=regularizers.l2(hp.Choice('l2_' + str(i), values=[1e-5, 1e-4]))))  # 缩小正则化范围
        model.add(layers.BatchNormalization())
        
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.3, max_value=0.5, step=0.1)  # 调整 Dropout 范围
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))  # 输出层

    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')  # 缩小学习率范围
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', rmse])
    return model

# 初始化 Keras Tuner 的 Bayesian Optimization
tuner = kt.BayesianOptimization(
    build_model,
    objective=Objective("val_rmse", direction="min"),
    max_trials=10,  # 减少试验次数
    directory='my_dir',
    project_name='bayesian_search_nn_optimized',
    overwrite=True,
    seed=SEED
)

# 定义回调函数
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_rmse', patience=5, restore_best_weights=True, mode='min'),  # 缩短耐心值
    keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.5, patience=3, min_lr=1e-6),          # 添加学习率调度
    keras.callbacks.TerminateOnNaN()
]

# 执行超参数搜索
tuner.search(
    X_train_scaled, y_train_scaled,
    epochs=50,        # 减少训练轮数
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)

# 获取最佳超参数
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("最佳超参数组合：")
print(f"隐藏层数量: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"第 {i+1} 层的神经元数量: {best_hps.get(f'units_{i}')}")
    print(f"第 {i+1} 层的激活函数: {best_hps.get('activation')}")
    print(f"第 {i+1} 层的 L2 正则化: {best_hps.get(f'l2_{i}')}")
    print(f"第 {i+1} 层的 Dropout 率: {best_hps.get(f'dropout_{i}')}")
print(f"学习率: {best_hps.get('learning_rate')}")

# 构建并训练最佳模型
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=50,        # 保持一致
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)

# 评估最佳模型
test_loss, test_mae, test_rmse = model.evaluate(X_test_scaled, y_test_scaled, verbose=2)
print('模型评估的均方误差 (MSE):', test_loss)
print('模型评估的均方根误差 (RMSE):', test_rmse)

# 反标准化和反对数转换
y_pred = model.predict(X_test_scaled)
y_pred_unscaled = y_scaler.inverse_transform(y_pred)
y_test_unscaled = y_scaler.inverse_transform(y_test_scaled)

y_pred_original = np.exp(y_pred_unscaled)
y_test_original = np.exp(y_test_unscaled)

# 评估原始空间
mse_original = mean_squared_error(y_test_original, y_pred_original)
rmse_original = np.sqrt(mse_original)
r2_original = r2_score(y_test_original, y_pred_original)

print(f'原始空间 - MSE: {mse_original}, RMSE: {rmse_original}, R2: {r2_original}')
