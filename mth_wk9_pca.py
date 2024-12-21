import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# Load the Excel file
file_path = "/Users/arsalonamini/Desktop/MTH5810/mth5810-dataset-for-PCA.xlsx" 
data = pd.read_excel(file_path)
print("Excel file loaded successfully.")
print(data.head)

# Display available columns
print("Available columns:")
print(data.columns)


# Step 1: Compute the mean of the dataset
xyz_data = data[['x', 'y', 'z']]
mean_vector = xyz_data.mean()
print("Mean vector:")
print(mean_vector)

# Step 2: Center the data by subtracting the mean
centered_data = xyz_data - mean_vector
print("Centered data:")
print(centered_data.head())

# Step 3: Calculate the covariance matrix
cov_matrix = np.cov(centered_data.T)
print("Covariance matrix:")
print(cov_matrix)

# Step 4: Find eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("Eigenvalues:")
print(eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Step 5: Project the data onto the two eigenvectors corresponding to the largest eigenvalues
# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Select the top 2 eigenvectors
top_2_eigenvectors = eigenvectors[:, :2]

# Project the data
projected_data = centered_data.dot(top_2_eigenvectors)
print("Projected 2D data:")
print(projected_data.head())

# Step 6: Visualization
# Plot original 3D data
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(xyz_data['x'], xyz_data['y'], xyz_data['z'], c='blue', label='Original Data')
ax.set_title('Original 3D Data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Plot projected 2D data
plt.subplot(122)
plt.scatter(projected_data.iloc[:, 0], projected_data.iloc[:, 1], c='red', label='Projected 2D Data')
plt.title('Projected 2D Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.tight_layout()
plt.show()






