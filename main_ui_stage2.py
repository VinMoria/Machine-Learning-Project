import tkinter as tk
from tkinter import ttk
import time
import os
import openpyxl
import json
import numpy as np
import pickle

import openpyxl.workbook

feature_label_list = []
feature_entry_list = []
entry_save_dict = {}
sector_feature_list = []
chosen_sector = ""

MODEL_PATH = "XGBoost_model/"

def destroy_entry_for_sector():
	global feature_label_list, feature_entry_list
	for label in feature_label_list:
		label.destroy()
	feature_label_list = []

	for entry in feature_entry_list:
		entry.destroy()
	feature_entry_list = []


def gen_entry_for_sector(event):
	global sector_feature_list
	global chosen_sector
	destroy_entry_for_sector()
	# read features for the sector from json file
	chosen_sector = sector_entry.get()
	with open("res.json", "r") as f:
		model_list = json.load(f)
	for model_dict in model_list:
		if model_dict["Sector"] == chosen_sector:
			sector_feature_list = model_dict["Features"]
			break

	# generate labels and entry for feature
	for index, feature in enumerate(sector_feature_list):
		# label
		new_label = tk.Label(root, text=feature)
		feature_label_list.append(new_label)
		new_label.grid(row=index // 4 * 2 + 2, column=index % 4, pady=5, padx=5)

		# entry
		new_entry = tk.Entry(root)
		feature_entry_list.append(new_entry)
		new_entry.grid(row=index // 4 * 2 + 3, column=index % 4, pady=5, padx=5)


	submit_button = tk.Button(root, text="Submit", command=onclick_submit)
	clear_button = tk.Button(root, text="Clear Entry", command=onclick_clear)
	submit_button.grid(row=1000,column=1,pady=5,padx=5)
	clear_button.grid(row=1000,column=2,pady=5,padx=5)


def onclick_submit():
	global entry_save_dict
	all_correct = True
	for index, entry in enumerate(feature_entry_list):
		input_str = entry.get()
		# check input
		if not check_input(input_str):
			all_correct = False
			feature_label_list[index].config(
				text=sector_feature_list[index] + " <Wrong Format>", fg="red")
		else:
			entry_save_dict[sector_feature_list[index]] = input_str
			feature_label_list[index].config(
				text=sector_feature_list[index], fg="black")
	# if all pass check
	if all_correct:
		# construct feature list for model
		input_feature_list = []
		for feature in sector_feature_list:
			if entry_save_dict[feature] == "":
				input_feature_list.append(np.nan)
			else:
				input_feature_list.append(float(entry_save_dict[feature]))

		# load model
		with open(f"{MODEL_PATH}{chosen_sector}.ml", "rb") as f:
			model = pickle.load(f)
		# predict value
		res_valuation = model.predict(np.array([input_feature_list]))[0]
		print(res_valuation)

		# write to excel
		workbook = openpyxl.Workbook()

		sheet1 = workbook.active
		sheet1.title = "Info&Valuation"

		sheet1["A1"] = "Company Name"
		sheet1["B1"] = company_name_entry.get()

		sheet1["A2"] = "Sector"
		sheet1["B2"] = chosen_sector

		for index, feature in enumerate(sector_feature_list):
			sheet1["A"+str(index+3)] = feature
			sheet1["B"+str(index+3)] = entry_save_dict[feature]

		sheet1["C1"] = "Valuation Result(M)"
		sheet1["D1"] = str(res_valuation)

		filename = "valuAItion_result_"+time.strftime('%Y-%m-%d_%H%M%S', time.localtime())+".xlsx"
		workbook.save(filename)
		os.startfile(filename)
		print(f"Successfully write into {filename}")

		#TODO 在UI上显示结果
			
def onclick_clear():
	global feature_label_list, feature_entry_list
	for index, label in enumerate(feature_label_list):
		label.config(
			text=sector_feature_list[index], fg="black")

	for entry in feature_entry_list:
		entry.delete(0, tk.END)

def check_input(in_str):
	# FIXME 是否可以接受空值
	if in_str == "" : return True
	try:
		float(in_str)
		return True
	except ValueError:
		return False


# Main start
SECTOR_LIST = ['Healthcare', 'Basic Materials', 'Financial', 'Consumer Defensive', 'Industrials',
               'Technology', 'Consumer Cyclical', 'Real Estate', 'Communication Services', 'Energy', 'Utilities']

root = tk.Tk()
root.title("valuAItion - Funder Version Prototype")

company_name_label = tk.Label(root, text="Company Name")
company_name_label.grid(row=0, column=1, pady=5, padx=5)

company_name_entry = tk.Entry(root)
company_name_entry.grid(row=1, column=1, pady=5, padx=5)

sector_label = tk.Label(root, text="Sector")
sector_label.grid(row=0, column=2, pady=5, padx=5)

sector_entry = ttk.Combobox(root, values=SECTOR_LIST)
sector_entry.grid(row=1, column=2, pady=5, padx=5)
sector_entry.bind("<<ComboboxSelected>>", gen_entry_for_sector)

root.mainloop()
