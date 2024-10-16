from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from datetime import datetime
import numpy as np
import pandas as pd
import json
import os
import pickle

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
	df_features.columns.values[0] = 'Feature'
	top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])
	X = df[top_features]

	# 划分训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	base_models = [
			('svm', SVR(kernel='rbf')),
			('rf', RandomForestRegressor(n_estimators=100, random_state=42))
	]
	
	stacked_model = StackingRegressor(estimators=base_models, final_estimator=RandomForestRegressor(n_estimators=100, random_state=42))
	# 创建管道，使用 SVM
	pipeline = Pipeline([
	('imputer', KNNImputer()),
	('scaler', StandardScaler()),
	('stacked', stacked_model)
	#('svm', SVR(kernel='rbf'))
	])

	# 定义参数网格
	param_grid = {
	'imputer__n_neighbors': [7, 10],  # KNN Imputer的邻居数量(3,5,7,10)
	'stacked__svm__C': [10],  # 使用更多点以细化C值范围(1, 10, 20)
	'stacked__svm__epsilon': np.linspace(0.01, 0.1, 10),  # 使用更多点以细化epsilon值范围
	'stacked__svm__gamma': [1, 10] ,# gamma的更细化取值范围['scale', 'auto'] + np.logspace(-3, 1, 5).tolist()
	'stacked__rf__n_estimators':[100]
}

# 使用GridSearchCV进行超参数搜索
	grid_search = GridSearchCV(
	pipeline,
	param_grid,
	scoring='neg_mean_squared_error',  # 使用负均方误差进行评分
	cv=10,  # 10折交叉验证
	verbose=2  # 输出详细的调参信息
)

# 训练模型并找到最佳参数
	grid_search.fit(X_train, y_train)

# 输出最佳参数组合
	print(f'Best parameters: {grid_search.best_params_}')
	best_model = grid_search.best_estimator_

# 预测测试集并计算RMSE
	y_test_pred = grid_search.best_estimator_.predict(X_test)
	test_mse = mean_squared_error(y_test, y_test_pred)
	test_rmse = np.sqrt(test_mse)

	print(f'Test MSE: {test_mse:.4f}')
	print(f'Test RMSE: {test_rmse:.4f}')
	

#保存模型
	saved_filename = f"{Sector}.ml"
	with open(f"SVM_model/{saved_filename}", "wb") as f:
		pickle.dump(best_model, f)
	print(f"save file: {saved_filename}")

	res_dict = {
		"Sector": Sector,
		"Features": top_features,
		"RMSE": test_rmse
	}
	return res_dict


##
# ==================  main start  ==================
# 清空SVM_model下的文件
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
	res_dict_list.append(train_svm_model(sector))

with open(f'SVM_model/res.json', 'w') as f:
	json.dump(res_dict_list, f, indent=4)

