import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import keras_tuner as kt
from kerastuner import HyperParameters, Objective
from kerastuner import Hyperband

# 固定随机种子以确保可复现性
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 数据读取和预处理
# -------------------- 数据读取和预处理 --------------------

# 定义文件路径和参数
FILEPATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20
Sector = 'Utilities'

# 读取主数据集
df = pd.read_csv(os.path.join(FILEPATH, f"{Sector}.csv"))

# 处理“Market Cap(M)”列：替换0为NaN，然后取自然对数
df["Market Cap(M)"] = np.log(df["Market Cap(M)"].replace(0, np.nan))

# 删除包含NaN的行（如果有的话）
df = df.dropna(subset=["Market Cap(M)"])

# 目标变量
y = df["Market Cap(M)"]

# 特征选择
df_features = pd.read_csv(os.path.join(FEATURE_PATH, f"{Sector}_feature_importance.csv"))
df_features.columns.values[0] = 'Feature'  # 确保第一列名为 'Feature'
top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])

# 特征矩阵
X = df[top_features].copy()  # 使用 .copy() 创建一个副本

# 检查特征矩阵中的 NaN 和无穷值
nan_features = X.columns[X.isna().any()].tolist()
inf_features = X.columns[np.isinf(X).any()].tolist()

print("包含 NaN 的特征:", nan_features)
print("包含无穷值的特征:", inf_features)

# 处理包含无穷值的特征
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# 初始化填充器，使用均值填充 NaN
imputer = SimpleImputer(strategy='mean')

# 对特征矩阵进行填充
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 确认是否还有 NaN 或无穷值
remaining_nan = X_imputed.isna().sum().sum()
remaining_inf = np.isinf(X_imputed.values).sum()

print("填充后特征矩阵中的 NaN 数量:", remaining_nan)
print("填充后特征矩阵中的无穷值数量:", remaining_inf)

# 移除目标变量中的极端值
lower_bound = y.quantile(0.01)
upper_bound = y.quantile(0.99)
y = y.clip(lower=lower_bound, upper=upper_bound)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=SEED
)

# 特征缩放（标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 标准化目标变量
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# 确认缩放后的数据中没有 NaN 或无穷值
print("缩放后的训练集是否包含 NaN:", np.isnan(X_train_scaled).any())
print("缩放后的训练集是否包含无穷值:", np.isinf(X_train_scaled).any())
print("缩放后的测试集是否包含 NaN:", np.isnan(X_test_scaled).any())
print("缩放后的测试集是否包含无穷值:", np.isinf(X_test_scaled).any())


# 定义超模型函数
def build_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1],)))

    for i in range(hp.Int('num_layers', 1, 3)):
        units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
        model.add(layers.Dense(units, activation='relu', 
                               kernel_regularizer=regularizers.l2(
                                   hp.Choice('l2_' + str(i), values=[1e-4, 1e-3, 1e-2]))))
        model.add(layers.BatchNormalization())
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1))  # 回归任务

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    # 定义 RMSE 指标
    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=[keras.metrics.MeanAbsoluteError(), rmse])

    return model

# 设置 Keras Tuner 并执行网格搜索
tuner = kt.GridSearch(
    build_model,
    objective=Objective("val_rmse", direction="min"),  # 确保这里使用正确的 RMSE 名称
    max_trials=1,
    directory='my_dir',
    project_name='grid_search_nn',
    overwrite=True,
    seed=SEED
)

# 执行超参数搜索
tuner.search(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
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

# 构建并训练最佳模型
model = tuner.hypermodel.build(best_hps)

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.TerminateOnNaN()
    ]
)

# 在测试集上评估模型
test_loss, test_mae, test_rmse = model.evaluate(X_test_scaled, y_test_scaled, verbose=2)
print('\n测试集的均方误差 (MSE):', test_loss)
print('测试集的平均绝对误差 (MAE):', test_mae)
print('测试集的均方根误差 (RMSE):', test_rmse)

# 预测并反标准化
predictions_scaled = model.predict(X_test_scaled[:5])
predictions_original = y_scaler.inverse_transform(predictions_scaled)
y_true_original = y_test.values.reshape(-1, 1)

for i in range(5):
    print(f"实际值: {y_true_original[i][0]:.4f}, 预测值: {predictions_original[i][0]:.4f}")

# 可视化训练过程
plt.figure(figsize=(18, 5))

# 损失（MSE）
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='训练损失 (MSE)')
plt.plot(history.history['val_loss'], label='验证损失 (MSE)')
plt.xlabel('Epoch')
plt.ylabel('均方误差 (MSE)')
plt.title('训练和验证损失')
plt.legend()

# 平均绝对误差 (MAE)
plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='训练 MAE')
plt.plot(history.history['val_mae'], label='验证 MAE')
plt.xlabel('Epoch')
plt.ylabel('平均绝对误差 (MAE)')
plt.title('训练和验证 MAE')
plt.legend()

# 均方根误差 (RMSE)
plt.subplot(1, 3, 3)
plt.plot(history.history['rmse'], label='训练 RMSE')
plt.plot(history.history['val_rmse'], label='验证 RMSE')
plt.xlabel('Epoch')
plt.ylabel('均方根误差 (RMSE)')
plt.title('训练和验证 RMSE')
plt.legend()

plt.tight_layout()
plt.show()
