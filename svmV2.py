from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

FILEPATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20

def train_svm_model(Sector):
    # 读取数据
    df = pd.read_csv(FILEPATH + Sector + ".csv")
    df["Market Cap(M)"] = np.log(df["Market Cap(M)"].replace(0, np.nan))
    y = df["Market Cap(M)"]
    
    # 特征选择
    df_features = pd.read_csv(FEATURE_PATH + Sector + "_feature_importance.csv")
    top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])
    X = df[top_features]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建管道，使用 SVM
    pipeline = Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', StandardScaler()),
        ('svm', SVR())
    ])

    # 定义参数网格，包括 KNNImputer 和 SVM 的超参数
    param_grid = {
        'imputer__n_neighbors': [3, 5, 7, 10],
        'svm__C': np.logspace(-3, 3, 7),           # 惩罚参数 C 的范围
        'svm__epsilon': np.linspace(0, 1, 5),      # epsilon 的范围
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # 核函数
    }

    # 网格搜索
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='neg_mean_squared_error',  # 使用负均方误差作为评分标准
        cv=5,                              # 5折交叉验证
        n_jobs=-1                          # 并行处理
    )
    grid_search.fit(X_train, y_train)

    # 输出最优参数
    best_n_neighbors = grid_search.best_params_['imputer__n_neighbors']
    best_C = grid_search.best_params_['svm__C']
    best_epsilon = grid_search.best_params_['svm__epsilon']
    best_kernel = grid_search.best_params_['svm__kernel']
    print(f'Best n_neighbors: {best_n_neighbors}')
    print(f'Best C: {best_C}')
    print(f'Best epsilon: {best_epsilon}')
    print(f'Best kernel: {best_kernel}')

    # 使用最优参数重新训练模型并评估
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')

# 调用函数
train_svm_model('Financial')
