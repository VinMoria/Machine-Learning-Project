import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

def show_importance():
    selected_sector = sector_combo.get()
    
    if not selected_sector:
        warning_label.config(text="Please select a sector", fg='red')
        return
    
    warning_label.config(text="")  # Clear previous warning
    
    try:
        # Load the saved feature importance
        importance = pd.read_csv(f"importance/{selected_sector}_feature_importance.csv")
        
        # Create a new window for showing important features
        importance_window = tk.Toplevel()
        importance_window.title("Saved Feature Importance")
        
        # Create ScrolledText to display important features
        feature_text = ScrolledText(importance_window, width=80, height=20)
        feature_text.pack()
        
        for index, row in importance.iterrows():
            feature_text.insert(tk.END, f"{row['Unnamed: 0']}: {row['Importance']}\n")
    
    except FileNotFoundError:
        warning_label.config(text="Feature importance file not found.", fg='red')

# Main application window
window = tk.Tk()
window.title("Sector Importance Display")

sectors = ["Basic Materials", "Communication Services", "Consumer Cyclical", 
           "Consumer Defensive", "Energy", "Financial", "Healthcare", 
           "Industrials", "Real Estate", "Technology", "Utilities"]

sector_combo = ttk.Combobox(window, values=sectors)
sector_combo.grid(row=0, column=1)
sector_label = tk.Label(window, text="Select Sector:")
sector_label.grid(row=0, column=0)

display_button = tk.Button(window, text="Display Importance", command=show_importance)
display_button.grid(row=1, columnspan=2)

warning_label = tk.Label(window, text="", fg='red')
warning_label.grid(row=2, columnspan=2)

window.mainloop()
