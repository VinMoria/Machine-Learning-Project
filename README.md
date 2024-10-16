# 模型运行流程
demo的逻辑：如果执行demo，只需要进行第5和第6步
注意：在实施步骤5和步骤6时，可能会花一点时间，请耐心等待。
1. 执行get_sector_ticker.py，获取每个行业下的所有ticker，存储到industry_stocks.csv
2. 执行train_set_gen.py，从industry_stocks.csv中，每个行业随机抽取200个ticker，获取数据，按行业存储到 /train_set 下
3. 分别运行 XGBoost_train.py, lightGBM_train.py, svmV2.py（基于2中获取的数据集）, 执行生成的模型(.ml文件)和res.json存储在对应的{}_model/ 下
4. 运行chose_model.py，读取以上模型的res.json，给每个行业选择最佳模型，生成新的res.json，并与最佳模型文件一起存储在best_model/ 下
5. 执行main_ui_stage2.py，根据输入的行业在best_model/中读取对应的模型，给出预测结果
6. 执行search_company.py，得到需要查询的公司的相关信息，新添加了词云和情感分数部分

# Model Running Process
Demo logic: If you need to execute the demo, only steps 5 and 6 are required. 
Note: It may take a while to implement step 5 and step 6, so please be patient.
1. Run get_sector_ticker.py to obtain all tickers for each sector and store them in industry_stocks.csv.
2. Run train_set_gen.py to randomly select 200 tickers from each sector listed in industry_stocks.csv, gather their data, and store it by sector in the /train_set folder.
3. Execute XGBoost_train.py, lightGBM_train.py, and svmV2.py (based on the dataset obtained in step 2), and store the generated models (.ml files) and res.json in the corresponding {}_model/ folder.
4. Run chose_model.py to read the res.json from the above models, select the best model for each sector, generate a new res.json, and store it along with the best model file in the best_model/ folder.
5. Execute main_ui_stage2.py to read the corresponding model from best_model/ based on the input sector and provide the prediction results.
6. Run search_company.py to retrieve relevant information about the company being queried, with the newly added sections for word cloud and sentiment score.