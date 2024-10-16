import pickle

def load_model(file_name):
	with open(file_name, "rb") as f:
		model = pickle.load(f)
	print(model.score)

load_model("XGBoost_model/Utilities_10101427.ml")