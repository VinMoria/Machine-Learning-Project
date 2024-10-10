from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import pandas as pd
import pickle
import numpy as np
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
	
	# 分割数据集为训练集和测试集
	scaler = MinMaxScaler()
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.2,
														random_state = 0)
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)		

	# way2 用交叉验证选择参数
 #rrrt
	r2 = []
	alpha_range = np.logspace(-2, 3, 100) #alpha 范围> 稳定

	for a in alpha_range:
		ridge = linear_model.Ridge(alpha = a)
		ridge_r2 = cross_val_score(ridge, X_train_scaled, y_train, cv=10).mean() #ridge，X_std自变量，y因变量，cv=10 10折交叉验证
		r2.append(ridge_r2)

	best_alpha = alpha_range[r2.index(max(r2))]
	print('best_alpha is ', best_alpha)
	print('best meanR2 is', max(r2))


	ridge_bestalpha = linear_model.Ridge(alpha = best_alpha)
	ridge_bestalpha.fit(X_train_scaled, y_train)
	ridge_trainscore = ridge_bestalpha.score(X_train_scaled, y_train)
	ridge_testscore = ridge_bestalpha.score(X_test_scaled, y_test)

		# 在验证集上进行预测
	y_pred = ridge_bestalpha.predict(X_test_scaled)

	mse = mean_squared_error(y_test, y_pred)
	rmse = mse**0.5
	print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}")

		# 保存模型
	saved_filename = f"{Sector}_{datetime.now().strftime('%m%d%H%M')}.ml"
	with open(f"XGBoost_model/{saved_filename}", "wb") as f:
		pickle.dump(ridge_bestalpha, f)
		print(f"save file: {saved_filename}")


# 清空XGBoost_model下的文件
folder_path = "Ridge_model"
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

# for sector in sector_list:
# 	train_XGBoost_for_sector(sector)

train_Ridge_for_sector("Utilities")