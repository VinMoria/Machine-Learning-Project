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
from sklearn.svm import SVR
import pandas as pd
import pickle
import os
import json


FILEPATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20

#get data
def train_SVM_for_sector(Sector):
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

	# 创建管道，使用 SVM
	pipeline = Pipeline([
        ('imputer', KNNImputer()),
        ('scaler', StandardScaler()),
        ('svm', SVR())  # 使用支持向量回归模型
    ])

    # 定义参数网格，包括 KNNImputer 和 SVM 的超参数
	param_grid = {
        'imputer__n_neighbors': [3, 5, 7, 10],
        'svm__C': np.logspace(-3, 3, 7),  # 惩罚参数 C 的范围
        'svm__epsilon': np.linspace(0, 1, 10),  # Epsilon 的范围
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

	rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

# 在 GridSearchCV 中使用自定义 RMSE 评分函数
	grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring=rmse_scorer,  # 使用 RMSE 作为评分标准
    cv=5, # 5折交叉验证
    verbose=2
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
	saved_filename = f"{Sector}_{datetime.now().strftime('%m%d%H%M')}.ml"
	with open(f"SVM_model/{saved_filename}", "wb") as f:
		pickle.dump(best_model, f)
	print(f"save file: {saved_filename}")

	res_dict = {
		"Sector": Sector,
		"Features": top_features,
		"RMSE": test_rmse 
	}
	return res_dict
# ==================  main start  ==================
# 清空XGBoost_model下的文件
folder_path = "SVM_model"
for filename in os.listdir(folder_path):
	file_path = os.path.join(folder_path, filename)
	if os.path.isfile(file_path):
		os.remove(file_path)
print("old models deleted")

sector_list = [
	"Healthcare",
	"Basic Materials",
	"Financial",
	"Consumer Defensive","Industrials",
	"Technology",
	"Consumer Cyclical",
	"Real Estate",
	"Communication Services",
	"Energy",
	"Utilities",
]

res_dict_list = []
for sector in sector_list:
	print(f"=============== {sector} start ===============")
	res_dict_list.append(train_SVM_for_sector(sector))

with open(f'SVM_model/res.json', 'w') as f:
    json.dump(res_dict_list, f, indent=4)
