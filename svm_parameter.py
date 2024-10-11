
import pickle
with open('SVM_model/Basic Materials.ml', 'rb') as f:
    data_load = pickle.load(f)
    print("Model loaded successfully.")
print(data_load)
