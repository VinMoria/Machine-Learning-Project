# 模型运行流程

1. 执行get_sector_ticker.py，获取每个行业下的所有ticker，存储到industry_stocks.csv
2. 执行train_set_gen.py，从industry_stocks.csv中，每个行业随机抽取200个ticker，获取数据，按行业存储到 /train_set 下。
3. 分别运行 XGBoost_train.py, lightGBM_train.py, svmV2.py（基于2中获取的数据集）, 执行生成的模型(.ml文件)和res.json存储在对应的{}_model/ 下
4. 运行chose_model.py，读取以上模型的res.json，给每个行业选择最佳模型，生成新的res.json，并与最佳模型文件一起存储在best_model/ 下
5. 执行main_ui_stage2.py，根据输入的行业在best_model/中读取对应的模型，给出预测结果