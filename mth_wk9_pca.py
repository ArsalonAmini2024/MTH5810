import pandas as pd
import matplotlib.pyplot as plt
import os


# Load the Excel file
file_path = "/Users/arsalonamini/Desktop/MTH5810/mth5810-dataset-for-PCA.xlsx" 
data = pd.read_excel(file_path)
print("Excel file loaded successfully.")
print(data.head)

# Display available columns
print("Available columns:")
print(data.columns)









