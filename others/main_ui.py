import tkinter as tk
from tkinter import ttk
import time
import os
import openpyxl

label_text_list = ["*Company Name(eg. apple):",
					"*Industry:",
					"*Cash flow - From (eg. 2024/Q3):",
					"*Cash flow - To (eg. 2024/Q3):",
					"Revenue:",
					"Cost:",
					"Selling Expenses:",
					"Administrative Expenses:",
					"R&D Expenses:",
					"Financial Expenses:",
					"Income tax:"]

inner_data_dic = {}
inner_data_key_list = ["name",
						"industry",
						"start_time",
						"end_time",
						"revenue",
						"cost",
						"selling_expenses",
						"administrative_expenses",
						"R&D_expenses",
						"financial_expenses",
						"income_tax",
						"net_profit",
						"gross_profit_margin",
						"cash_flow_list"]

input_rule_list = [1, 2, 3, 3, 5, 5, 5, 5, 5, 5, 5]

label_list = []
entry_list = []
cash_flow_entry_list = []
cash_flow_label_list = []


yahoo_list= [
	"Basic Materials",
	"Communication Services",
	"Consumer Cyclical",
	"Consumer Defensive",
	"Energy",
	"Financials",
	"Healthcare",
	"Industrials",
	"Technology",
	"Utilities"
]

# submit button
def submit_input():
	all_correct = True

	# Check Input and show Format Message
	for index, entry in enumerate(entry_list):
		input_str = entry.get()
		inner_data_dic[inner_data_key_list[index]] = input_str
		if not check_input(input_str, input_rule_list[index]):
			all_correct = False
			label_list[index].config(text = label_text_list[index]+" <Wrong Format>", fg="red")

	# Check cash flow opened
	if len(cash_flow_entry_list) == 0:
		all_correct = False
		cash_flow_label.config(text = "Empty Cash Flow", fg="red")

	# Check cash flow input
	cash_flow_value_list = []
	for index, enrty in enumerate(cash_flow_entry_list):
		input_str = enrty.get()
		if check_input(input_str, 4):
			if len(input_str)>0:
				cash_flow_value_list.append(float(input_str))
			else:
				cash_flow_value_list.append("")
		else:
			all_correct = False
			cash_flow_label_list[index].config(text=cash_flow_label_list[index].cget("text")[:7]+" <Wrong Format>",fg="red")

	print(inner_data_dic)
	if all_correct:
		inner_data_dic["cash_flow_list"] = cash_flow_value_list

		inner_data_dic["net_profit"] = get_net_profit()
		inner_data_dic["gross_profit_margin"] = get_gross()

		filename = "saved_data_"+time.strftime('%Y-%m-%d_%H%M%S', time.localtime())+".xlsx"
		write_data_to_excel(inner_data_dic,filename)
		os.startfile(filename)
		cash_flow_label.config(text="Your Input has been saved as\n"
						+"<"+filename+">\n"+
						"Click Clear Entry to start a new one")
		summary_label.config(text="Summary:"+"\nCompany Name: "+ inner_data_dic["name"]+"\nIndustry: "+ inner_data_dic["industry"]+"\nNet Profit: " + inner_data_dic["net_profit"]+"\nGross Profit Margin: " + inner_data_dic["gross_profit_margin"])

def get_net_profit():
	for i in range(4, 11):
		if len(inner_data_dic[inner_data_key_list[i]])==0:
			return ""
	net = float(inner_data_dic["revenue"])-float(inner_data_dic["cost"])-float(inner_data_dic["selling_expenses"])-float(inner_data_dic["administrative_expenses"])-float(inner_data_dic["R&D_expenses"])-float(inner_data_dic["financial_expenses"])-float(inner_data_dic["income_tax"])
	return str(net)

def get_gross():
	if len(inner_data_dic["revenue"]) == 0 or len(inner_data_dic["cost"])==0 or float(inner_data_dic["revenue"]) == 0:
		return ""
	else:
		gross = round(((float(inner_data_dic["revenue"])-float(inner_data_dic["cost"]))/float(inner_data_dic["revenue"]))*100, 2)
		return str(gross) + "%"

def write_data_to_excel(data, filename):	 
	def generate_quarters(start_time, end_time):
		start_year, start_quarter = map(int, start_time.split("/Q"))
		end_year, end_quarter = map(int, end_time.split("/Q"))
		  
		quarters = []
		  
		year, quarter = start_year, start_quarter
		while (year < end_year) or (year == end_year and quarter <= end_quarter):
			quarters.append(f"{year}/Q{quarter}")
			quarter += 1
			if quarter > 4:
				quarter = 1
				year += 1
		return quarters

	quarters = generate_quarters(data["start_time"], data["end_time"])

	workbook = openpyxl.Workbook()

	sheet1 = workbook.active
	sheet1.title = "Company Info"

	sheet1["A1"] = "Key"
	sheet1["B1"] = "Value"

	for i in range(13):
		sheet1["A"+str(i+2)] = inner_data_key_list[i]
		sheet1["B"+str(i+2)] = data[inner_data_key_list[i]]

	# sheet 2
	sheet2 = workbook.create_sheet(title="Cash Flow")

	sheet2["A1"] = "Quarter"
	sheet2["B1"] = "Cash Flow"

	for idx, (quarter, cash_flow) in enumerate(zip(quarters, data["cash_flow_list"]), start=2):
		sheet2[f"A{idx}"] = quarter
		sheet2[f"B{idx}"] = cash_flow

	workbook.save(filename)
	print(f"Successfully write into {filename}")

# Clear input content
def clear_input():
	global entry_list
	for entry in entry_list:
		entry.delete(0, tk.END)

	global cash_flow_entry_list
	for entry in cash_flow_entry_list:
		entry.destroy()
	cash_flow_entry_list = []

	global cash_flow_label_list
	for label in cash_flow_label_list:
		label.destroy()
	cash_flow_label_list = []

	for index, label in enumerate(label_list):
		label.config(text=label_text_list[index], fg="black")

	cash_flow_label.config(text = "")
	summary_label.config(text="")

# format check with rule
def check_input(in_str,rule_no):
	if rule_no == 0:
		return True
	elif rule_no == 1: # not empty
		return len(in_str) > 0
	elif rule_no == 2: # Yahoo list
		return in_str in yahoo_list
	elif rule_no == 3: # quarter format
		passcheck = True
		year_str = in_str[0:4]
		mid_str = in_str[4:6]
		quar_str = in_str[6:7]
		if len(in_str) != 7:
			passcheck = False
			print(len(in_str))
		if mid_str != "/Q":
			passcheck = False
			print(mid_str)
		if not (year_str.isdigit() and quar_str.isdigit()):
			passcheck = False
		return passcheck
	elif rule_no == 4: # float
		try:
			float(in_str)
			return True
		except ValueError:
			return False
	elif rule_no ==5: # float optional
		if len(in_str) == 0:
			return True
		else:
			try:
				float(in_str)
				return True
			except ValueError:
				return False


def gen_cash_flow_entry():
	for i in range(2,4):
		label_list[i].config(text = label_text_list[i], fg="black")
	# check start and end
	global cash_flow_entry_list
	for entry in cash_flow_entry_list:
		entry.destroy()
	cash_flow_entry_list = []

	global cash_flow_label_list
	for label in cash_flow_label_list:
		label.destroy()
	cash_flow_label_list = []

	all_correct = True
	for i in range(2,4):
		input_str = entry_list[i].get()
		inner_data_dic[inner_data_key_list[i]] = input_str
		if not check_input(input_str,3):
			all_correct = False
			label_list[i].config(text = label_text_list[i]+" <Wrong Format>", fg="red")

	# Create more Entry for cash flow
	if all_correct:
		cash_flow_label.config(text = "")
		start_stamp = int(inner_data_dic["start_time"][0:4])*4 + int(inner_data_dic["start_time"][6:7])
		end_stamp = int(inner_data_dic["end_time"][0:4])*4 + int(inner_data_dic["end_time"][6:7])
		cash_flow_num = end_stamp - start_stamp + 1
		if cash_flow_num>0:
			for i in range(cash_flow_num):
					label_text = "*" + str((start_stamp+i-1)//4)+"/Q"+str((start_stamp+i-1)%4+1)
					new_label = tk.Label(root, text=label_text)
					cash_flow_label_list.append(new_label)
					new_label.grid(row=100+(i//4)*2,column=i%4,pady=5,padx=5)

					new_entry = tk.Entry(root)
					cash_flow_entry_list.append(new_entry)
					new_entry.grid(row=100+(i//4)*2+1,column=i%4,pady=5,padx=5)

# Main start
root = tk.Tk()
root.title("valuAItion - Funder Version Prototype")


for index, label_text in enumerate(label_text_list):
	# input hint
	new_label = tk.Label(root, text=label_text)
	label_list.append(new_label)
	new_label.grid(row=index//4*2,column=index%4,pady=5,padx=5)

	# input box
	if index == 1:
		new_entry = ttk.Combobox(root, values=yahoo_list)
		new_entry.grid(row=index//4*2+1,column=index%4,pady=5,padx=5)
		entry_list.append(new_entry)
	else:
		new_entry = tk.Entry(root)
		entry_list.append(new_entry)
		new_entry.grid(row=index//4*2+1,column=index%4,pady=5,padx=5)

# gen_cash_flow button
quar_button = tk.Button(root, text="Input Cash Flow", command=gen_cash_flow_entry)
# cash flow label
cash_flow_label = tk.Label(root, text="")
# summary label
summary_label = tk.Label(root, text="")
# submit button
submit_button = tk.Button(root, text="Submit", command=submit_input)
# clear button
clear_button = tk.Button(root, text="Clear Entry", command=clear_input)

quar_button.grid(row=1000,column=0,pady=5,padx=5)
cash_flow_label.grid(row=1000,column=1,pady=5,padx=5)
summary_label.grid(row=1000,column=2,columnspan=2,pady=5,padx=5)
submit_button.grid(row=1001,column=1,pady=5,padx=5)
clear_button.grid(row=1001,column=2,pady=5,padx=5)

root.mainloop()