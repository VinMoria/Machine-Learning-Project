import json
import shutil
import os

MODEL_PATH_LIST = [
	"XGBoost_model/",
	"SVM_model/",
	"lightGBM_model/"
]

BEST_MODEL_PATH = "best_model/"

SECTOR_LIST = [
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

for filename in os.listdir(BEST_MODEL_PATH):
	file_path = os.path.join(BEST_MODEL_PATH, filename)
	if os.path.isfile(file_path):
		os.remove(file_path)
print("old models deleted")

json_list = []

for model_path in MODEL_PATH_LIST:
	with open(model_path+"res.json","r") as f:
		json_list.append(json.load(f))

best_model_list_for_json = []

for sector in SECTOR_LIST: # 遍历sector
	sector_list_from_all_model = []
	for one_model_list in json_list: # 获取每个模型下对应sector的RMSE用于比较
		for dict_data in one_model_list: # 遍历搜索这个模型下对应sector的dict
			if dict_data["Sector"] == sector:
				sector_list_from_all_model.append(dict_data)
				break

	min_index = 0
	min_rmse = sector_list_from_all_model[0]["RMSE"]
	for index, sector_dict in enumerate(sector_list_from_all_model):
		if sector_dict["RMSE"] < min_rmse:
			min_index = index
			min_rmse = sector_dict["RMSE"]

	best_model_list_for_json.append(sector_list_from_all_model[min_index])
	# TODO 复制model文件到best_model中
	shutil.copy(f"{MODEL_PATH_LIST[min_index]}{sector}.ml", f"{BEST_MODEL_PATH}{sector}.ml")

	print(f"Sector: {sector}, Choose: {MODEL_PATH_LIST[min_index]}")

with open(f"{BEST_MODEL_PATH}res.json", "w") as f:
	json.dump(best_model_list_for_json, f, indent=4)
