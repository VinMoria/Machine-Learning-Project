import tkinter as tk
from tkinter import ttk
import time
import os
import openpyxl
import json

feature_label_list = []
feature_entry_list = []
entry_save_dict = []

def destroy_entry_for_sector():
	global feature_label_list, feature_entry_list
	for label in feature_label_list:
		label.destroy()
	feature_label_list = []

	for entry in feature_entry_list:
		entry.destroy()
	feature_entry_list = []

def gen_entry_for_sector(event):
	# read features for the sector from json file
	destroy_entry_for_sector()

	sector = sector_entry.get()
	sector_feature_list = []
	with open("res.json", "r") as f:
		model_list = json.load(f)
	for model_dict in model_list:
		if model_dict["Sector"] == sector:
			sector_feature_list = model_dict["Features"]
			break
	
	# genrate labels and entry for feature
	for index, feature in enumerate(sector_feature_list):
		# label
		new_label = tk.Label(root, text=feature)
		feature_label_list.append(new_label)
		new_label.grid(row=index//4*2+2,column=index%4,pady=5,padx=5)

		# entry
		new_entry = tk.Entry(root)
		feature_entry_list.append(new_entry)
		new_entry.grid(row=index//4*2+3,column=index%4,pady=5,padx=5)

def onclick_submit():
	all_correct = True
	for index, entry in enumerate(feature_entry_list):
		input_str = entry.get()
		entry_save_dict[feature_label_list[index]] = input_str
		if not check_input(input_str):
			all_correct = False
			feature_label_list[index].config(text = feature_label_list+" <Wrong Format>", fg="red")

def check_input(in_str):
	return True

# Main start
SECTOR_LIST = ['Healthcare', 'Basic Materials', 'Financial', 'Consumer Defensive','Industrials', 'Technology', 'Consumer Cyclical', 'Real Estate','Communication Services', 'Energy', 'Utilities']

root = tk.Tk()
root.title("valuAItion - Funder Version Prototype")

company_name_label = tk.Label(root, text="Company Name")
company_name_label.grid(row=0,column=1,pady=5,padx=5)

company_name_entry = tk.Entry(root)
company_name_entry.grid(row=1,column=1,pady=5,padx=5)

sector_label = tk.Label(root, text="Sector")
sector_label.grid(row=0,column=2,pady=5,padx=5)

sector_entry = ttk.Combobox(root, values=SECTOR_LIST)
sector_entry.grid(row=1,column=2,pady=5,padx=5)
sector_entry.bind("<<ComboboxSelected>>", gen_entry_for_sector)



root.mainloop()