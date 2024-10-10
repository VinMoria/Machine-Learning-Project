from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn import linear_model
import pandas as pd
import pickle
import numpy as np
import os

FILEPATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 10

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
	#scaler = MinMaxScaler()
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.2,
														random_state = 0)
	#X_train		 = scaler.fit_transform(X_train)
	#X_test		 = scaler.transform(X_test)		

	imputer = KNNImputer(n_neighbors=5)

# Fit the imputer on training data and transform both training and test data
	X_train_imputed = imputer.fit_transform(X_train)
	X_test_imputed = imputer.transform(X_test)

	# way2 用交叉验证选择参数
	r2 = []
	alpha_range = np.logspace(-2, 3, 100) #alpha 范围> 稳定
	for a in alpha_range:
		ridge = linear_model.Ridge(alpha = a)
		ridge_r2 = cross_val_score(ridge, X_train_imputed, y_train, cv=10).mean() #ridge，X_std自变量，y因变量，cv=10 10折交叉验证
		r2.append(ridge_r2)
    
	best_alpha = alpha_range[r2.index(max(r2))]

	ridge_bestalpha = linear_model.Ridge(alpha = best_alpha)
	ridge_bestalpha.fit(X_train_imputed, y_train)
	ridge_trainscore = ridge_bestalpha.score(X_train_imputed, y_train)
	ridge_testscore = ridge_bestalpha.score(X_test_imputed, y_test)

		# 在验证集上进行预测
	y_pred = ridge_bestalpha.predict(X_test_imputed)

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

train_Ridge_for_sector("Utilities")



# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import ElasticNet
# from sklearn.impute import KNNImputer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import make_scorer, mean_squared_error

# # 创建示例数据，包含一些缺失值
# np.random.seed(42)
# X = np.random.rand(100, 5)  # 100行，5列的特征
# y = X @ np.array([1.5, -2, 3, 0, -1]) + np.random.randn(100) * 0.5  # 随机生成目标变量

# # 将部分数据设置为NaN
# X[0:10, 0] = np.nan

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 创建一个Pipeline，其中包括KNNImputer和ElasticNet
# pipeline = Pipeline([
#     ('imputer', KNNImputer()),
#     ('elastic_net', ElasticNet(max_iter=10000, random_state=42))
# ])

# # 定义参数网格，包括KNNImputer的n_neighbors和ElasticNet的超参数
# param_grid = {
#     'imputer__n_neighbors': [3, 5, 7, 10],  # 选择不同的n_neighbors值
#     'elastic_net__alpha': np.logspace(-3, 3, 10),  # alpha的范围
#     'elastic_net__l1_ratio': np.linspace(0, 1, 5)  # l1_ratio的范围
# }

# # 使用GridSearchCV进行超参数搜索，使用neg_mean_squared_error作为评分标准
# grid_search = GridSearchCV(
#     pipeline,
#     param_grid,
#     scoring=make_scorer(mean_squared_error, greater_is_better=False),  # 使用MSE作为评分标准
#     cv=5  # 5折交叉验证
# )
# grid_search.fit(X_train, y_train)

# # 获取最优的n_neighbors、alpha和l1_ratio
# best_n_neighbors = grid_search.best_params_['imputer__n_neighbors']
# best_alpha = grid_search.best_params_['elastic_net__alpha']
# best_l1_ratio = grid_search.best_params_['elastic_net__l1_ratio']
# print(f'Best n_neighbors: {best_n_neighbors}')
# print(f'Best alpha: {best_alpha}')
# print(f'Best l1_ratio: {best_l1_ratio}')

# # 使用最优参数重新训练模型，并在测试集上评估
# best_model = grid_search.best_estimator_
# y_test_pred = best_model.predict(X_test)
# test_mse = mean_squared_error(y_test, y_test_pred)
# test_rmse = np.sqrt(test_mse)

# print(f'Test MSE: {test_mse:.4f}')
# print(f'Test RMSE: {test_rmse:.4f}')