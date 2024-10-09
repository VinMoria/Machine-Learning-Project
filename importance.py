import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

# Function to calculate and save feature importance
def calculate_importance(sector):
    df = pd.read_csv(f"train_set/{sector}.csv")
    
    X = df.iloc[:, 2:].drop(columns=['Market Cap(M)'])  # Exclude the first two columns and y
    y = df['Market Cap(M)']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # xgboost model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame(importances, index=X.columns, columns=['Importance'])
    
    # Sort and save as csv
    feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
    feature_importance.to_csv(f"importance/{sector}_feature_importance.csv", index=True)

    return feature_importance

# Function to create GUI and display the features
def show_importance():
    selected_sector = sector_combo.get()
    
    if not selected_sector:
        warning_label.config(text="Please select a sector", fg='red')
        return
    
    warning_label.config(text="")  # Clear previous warning
    
    importance = calculate_importance(selected_sector)
    
    # Create a new window for showing important features
    importance_window = tk.Toplevel()
    importance_window.title("Feature Importance")
    
    # Create ScrolledText to display important features
    feature_text = ScrolledText(importance_window, width=80, height=20)
    feature_text.pack()
    
    for index, row in importance.iterrows():
        feature_text.insert(tk.END, f"{index}: {row['Importance']}\n")

# Main application window
window = tk.Tk()
window.title("Sector Importance Calculator")

sectors = ["Basic Materials", "Communication Services", "Consumer Cyclical", 
           "Consumer Defensive", "Energy", "Financial", "Healthcare", 
           "Industrials", "Real Estate", "Technology", "Utilities"]

# drop-down list
sector_combo = ttk.Combobox(window, values=sectors)
sector_combo.grid(row=0, column=1)
sector_label = tk.Label(window, text="Select Sector:")
sector_label.grid(row=0, column=0)

calculate_button = tk.Button(window, text="Calculate Importance", command=show_importance)
calculate_button.grid(row=1, columnspan=2)

warning_label = tk.Label(window, text="", fg='red')
warning_label.grid(row=2, columnspan=2)

window.mainloop()
