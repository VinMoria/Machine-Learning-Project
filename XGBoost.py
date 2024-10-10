import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

TRAIN_SET_PATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20

# TODO Features select

# def clean_outlier(df):
#	 print(df.shape)
#	 # 计算 Q1 和 Q3
#	 Q1 = df["Market Cap(M)"].quantile(0.25)
#	 Q3 = df["Market Cap(M)"].quantile(0.75)
#	 IQR = Q3 - Q1

#	 # 定义异常值的上下界
#	 lower_bound = Q1 - 1.5 * IQR
#	 upper_bound = Q3 + 1.5 * IQR

#	 # 筛选出在正常范围内的行
#	 df_cleaned = df[
#		 (df["Market Cap(M)"] >= lower_bound) & (df["Market Cap(M)"] <= upper_bound)
#	 ]
#	 print(df_cleaned.shape)
#	 return df_cleaned


def train_XGBoost_for_sector(Sector):
	# 假设 X 是特征矩阵，y 是目标变量
	df = pd.read_csv(TRAIN_SET_PATH + Sector + ".csv")

	# log 方法
	df["Market Cap(M)"] = np.log(df["Market Cap(M)"].replace(0, np.nan))
	y = df["Market Cap(M)"]

	# 读取选取的特征值
	df_features = pd.read_csv(FEATURE_PATH + Sector + "_feature_importance.csv")
	df_features.columns.values[0] = 'Feature'
	top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])

	X = df[top_features]
	print(X.shape)

	# Normalize
	# scaler = MinMaxScaler()
	# X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
	# print(list(X.iloc[0]))

	# 分割数据集为训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)
	print(y_test.describe())

	# 定义参数网格
	param_grid = {
		"max_depth": [3, 4, 5, 6],
		"eta": [0.01, 0.1, 0.2],
		"subsample": [0.5, 0.7, 1.0],
		"colsample_bytree": [0.5, 0.7, 1.0],
	}

	# 创建 XGBoost 回归模型
	model = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse")

	# 使用 GridSearchCV 进行参数搜索
	grid_search = GridSearchCV(
		model, param_grid, scoring="neg_mean_squared_error", cv=5, verbose=1
	)
	grid_search.fit(X_train, y_train)

	# 输出最佳参数
	print("Best parameters found: ", grid_search.best_params_)

	# 使用最佳参数训练最终模型
	best_model = grid_search.best_estimator_

	# 在验证集上进行预测
	y_pred = best_model.predict(X_test)

	mse = mean_squared_error(y_test, y_pred)
	rmse = mse**0.5
	print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}")

	# 保存模型
	saved_filename = f"{Sector}_{datetime.now().strftime('%m%d%H%M')}.ml"
	with open(f"XGBoost_model/{saved_filename}", "wb") as f:
		pickle.dump(best_model, f)
	print(f"save file: {saved_filename}")

	res_dict = {
		"Sector": Sector,
		"Features": top_features,
		"RMSE": rmse
	}
	return res_dict



# ==================  main start  ==================
# 清空XGBoost_model下的文件
folder_path = "XGBoost_model"
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
	res_dict_list.append(train_XGBoost_for_sector(sector))

with open(f'XGBoost_model/res.json', 'w') as f:
    json.dump(res_dict_list, f, indent=4)

# test from web
