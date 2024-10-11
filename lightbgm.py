import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
import pickle
import json

# 数据读取和预处理
FILEPATH = "train_set/"
FEATURE_PATH = "importance/"
FEATURE_SELECT_TOP = 20



def train_lightBGM_for_sector(Sector):
    # 读取主数据集
    df = pd.read_csv(os.path.join(FILEPATH, f"{Sector}.csv"))

    # 处理“Market Cap(M)”列：替换0为NaN，然后取自然对数
    df["Market Cap(M)"] = df["Market Cap(M)"].replace(0, np.nan)
    df["Market Cap(M)"] = np.log(df["Market Cap(M)"])
    #df["Market Cap(M)"].fillna(df["Market Cap(M)"].mean(), inplace=True)  # 填补NaN

    # 目标变量
    y = df["Market Cap(M)"]

    # 特征选择
    df_features = pd.read_csv(os.path.join(FEATURE_PATH, f"{Sector}_feature_importance.csv"))
    df_features.columns.values[0] = 'Feature'
    top_features = list(df_features.sort_values(by='Importance', ascending=False).head(FEATURE_SELECT_TOP)["Feature"])

    # 特征矩阵
    X = df[top_features]

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 特征缩放（标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 检查是否需要对目标变量进行缩放
    # LightGBM 不需要对目标变量进行缩放，保持 y 为原始的对数变换后的值

    # 定义 LightGBM 回归器
    lgb_reg = lgb.LGBMRegressor(
        objective='regression',
        random_state=42,
        n_jobs=-1
    )

    # 定义超参数搜索空间
    param_dist = {
        'num_leaves': randint(20, 150),
        'max_depth': randint(3, 15),
        'learning_rate': uniform(0.01, 0.3),  # 从0.01到0.31
        'n_estimators': randint(100, 1000),
        'subsample': uniform(0.5, 0.5),  # 0.5到1.0
        'colsample_bytree': uniform(0.5, 0.5),  # 0.5到1.0
        'min_child_samples': randint(5, 100)
    }

    # 初始化 RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=lgb_reg,
        param_distributions=param_dist,
        n_iter=50,  # 试验次数，根据需要调整
        scoring='neg_mean_squared_error',
        cv=3,  # 交叉验证折数
        verbose=1,
        random_state=SEED,
        n_jobs=-1
    )

    # 执行超参数搜索
    random_search.fit(X_train_scaled, y_train)

    # 获取最佳参数
    best_params = random_search.best_params_
    print("最佳参数组合：")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # 使用最佳参数训练最终模型
    best_lgb_reg = random_search.best_estimator_

    # 定义评估集
    eval_set = [(X_test_scaled, y_test)]

    # 训练最佳模型，并使用早停策略
    best_lgb_reg.fit(
        X_train_scaled, y_train,
        eval_set=eval_set,
        eval_metric='mse',  # 可根据需要选择其他评估指标
    )

    # 预测
    y_pred = best_lgb_reg.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}")
    
    # 保存模型
    saved_filename = f"{Sector}.ml"
    with open(f"lightBGM_model/{saved_filename}", "wb") as f:
        pickle.dump(best_lgb_reg, f)
    print(f"save file: {saved_filename}")

    res_dict = {
		"Sector": Sector,
		"Features": top_features,
		"RMSE": rmse
	}
    return res_dict



# ==================  main start  ==================
# 清空XGBoost_model下的文件
folder_path = "lightBGM_model"
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
	res_dict_list.append(train_lightBGM_for_sector(sector))

with open(f'lightBGM_model/res.json', 'wb') as f:
    json.dump(res_dict_list, f, indent=4)
