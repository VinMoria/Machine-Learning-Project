import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 假设 X 是特征矩阵，y 是目标变量
X, y = some_regression_dataset()  # 替换为你的数据集加载方法

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  # 回归问题
    'max_depth': 4,
    'eta': 0.1,
    'eval_metric': 'rmse'  # 使用均方根误差作为评估指标
}

# 定义验证集
evals = [(dtest, 'test')]

# 训练模型
model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)

y_pred = model.predict(dtest)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}')
