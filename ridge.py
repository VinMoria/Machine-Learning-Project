import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn import linear_model
import pandas as pd
import pickle
import os


FILEPATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20

#get data
def train_Ridge_for_sector(Sector):
	# 假设 X 是特征矩阵，y 是目标变量
	df = pd.read_csv(FILEPATH + Sector + ".csv")
	# log 方法
	df["Market Cap(M)"] = np.log(df["Market Cap(M)"].replace(0, np.nan))
	y = df["Market Cap(M)"]
 
	df_features = pd.read_csv(FEATURE_PATH + Sector + "_feature_importance.csv")
	df_features.columns.values[0] = 'Feature'
	top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])

	X = df[top_features]
	


# 划分训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	pipeline = Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', StandardScaler()),
        ('ridge', linear_model.Ridge(max_iter=20000, random_state=42))  # 使用 Ridge 回归
    ])

    # 定义参数网格，包括 KNNImputer 和 Ridge 的超参数
	param_grid = {
        'imputer__n_neighbors': [3, 5, 7, 10],  # KNN 的邻居数量
        'ridge__alpha': np.logspace(-3, 3, 10)  # Ridge的 alpha 范围
    }


# 使用GridSearchCV进行超参数搜索，使用neg_mean_squared_error作为评分标准
	grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring=make_scorer(mean_squared_error, greater_is_better=False),  # 使用MSE作为评分标准
    cv=5  # 5折交叉验证
)
	grid_search.fit(X_train, y_train)

# 获取最优的n_neighbors、alpha和l1_ratio
	best_n_neighbors = grid_search.best_params_['imputer__n_neighbors']
	best_alpha = grid_search.best_params_['ridge__alpha']
	
	print(f'Best n_neighbors: {best_n_neighbors}')
	print(f'Best alpha: {best_alpha}')


# 使用最优参数重新训练模型，并在测试集上评估
	best_model = grid_search.best_estimator_
	y_test_pred = best_model.predict(X_test)
	test_mse = mean_squared_error(y_test, y_test_pred)
	test_rmse = np.sqrt(test_mse)

	print(f'Test MSE: {test_mse:.4f}')
	print(f'Test RMSE: {test_rmse:.4f}')
 

train_Ridge_for_sector("Utilities")