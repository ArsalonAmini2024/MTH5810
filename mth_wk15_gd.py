import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def compute_cost(X, y, theta):
    """
    Compute the cost function for linear regression using mean squared error.
    
    Parameters:
        X (ndarray): Feature matrix of shape (m, n)
        y (ndarray): Target values of shape (m, )
        theta (ndarray): Parameter vector of shape (n, )
        
    Returns:
        float: The mean squared error cost.
    """
    m = len(y)  # number of training examples
    predictions = X.dot(theta)
    sq_errors = (predictions - y) ** 2
    cost = (1 / (2 * m)) * np.sum(sq_errors)
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Perform gradient descent to learn theta parameters.
    
    Parameters:
        X (ndarray): Feature matrix (m x n)
        y (ndarray): Target vector (m, )
        theta (ndarray): Initial parameter vector (n, )
        alpha (float): Learning rate
        num_iters (int): Number of iterations
        
    Returns:
        theta (ndarray): The optimized parameters.
        cost_history (list): The history of cost values for each iteration.
    """
    m = len(y)
    cost_history = []
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = predictions - y
        gradient = (1/m) * X.T.dot(error)
        theta = theta - alpha * gradient
        
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

def predict(X_new, theta, X_mean, X_std):
    """
    Make predictions using learned parameters (theta).
    
    Parameters:
        X_new (ndarray): New input features of shape (m_new, n) without the bias column.
        theta (ndarray): Learned parameters of shape (n+1, ).
        X_mean (ndarray): Mean of each feature from the training set.
        X_std (ndarray): Std of each feature from the training set.
        
    Returns:
        predictions (ndarray): Predicted values of shape (m_new, ).
    """
    # Normalize the new input data using the same training stats
    X_norm = (X_new - X_mean) / X_std
    
    # Add column of ones for the intercept
    X_b_new = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]
    
    # Predict
    predictions = X_b_new.dot(theta)
    return predictions


def main():
    
    file_path = "/Users/arsalonamini/Desktop/MTH5810/multi_feature_dataset_100_samples.csv"
    data = pd.read_csv(file_path, header=None)
    print("Excel file loaded successfully.")
    
    # Assign column names
    data.columns = ["Square Footage", "Number of Bedrooms", "Age of House", "House Price"]
    data.info()
    print(data.head())
    
    # Extract features (X) the first three columns and target (y) last column
    X = data[["Square Footage", "Number of Bedrooms", "Age of House"]].values
    y = data["House Price"].values
    
    # Compute mean and std for each feature
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    # Normalize the features
    X_norm = (X - X_mean) / X_std
    
    # Add a column of ones to X to account for the bias (intercept) term
    X_b = np.c_[np.ones((len(y), 1)), X_norm]  # shape (m, n+1)
    
    # Initialize theta (parameters) to zeros
    theta = np.zeros(X_b.shape[1])
    
    # Set learning rate (alpha) and number of iterations (num_iters)
    alpha = 1e-4
    num_iters = 10000
    
    # Run gradient descent
    theta, cost_history = gradient_descent(X_b, y, theta, alpha, num_iters)
    
    # Output the optimized parameters
    print("Optimized parameters (theta):")
    print(theta)
    
    # Plot the cost history to visualize convergence
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_iters+1), cost_history, 'b-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost J')
    plt.title('Cost Function History')
    plt.grid(True)
    plt.show()
    
    print("About to plot alpha experiments...")
    feature_names = ["Intercept", "Square Footage", "Number of Bedrooms", "Age of House"]
    # Experimentation: Try different learning rates
    for test_alpha in [1e-2, 1e-4, 5e-6]:
        test_theta = np.zeros(X_b.shape[1])
        test_theta, test_cost_history = gradient_descent(X_b, y, test_theta, test_alpha, num_iters)
        print(f"\nAlpha: {test_alpha}")
        print("Final Cost:", test_cost_history[-1])
        print("Final Theta:", test_theta)
        for name, val in zip(feature_names, test_theta):
            print(f"{name}: {val}")
        
        plt.plot(range(1, num_iters+1), test_cost_history, label=f'alpha={test_alpha}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iterations for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Compare with scikit-learn's LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    
    print("\nComparison with scikit-learn's LinearRegression:")
    print("Scikit-learn intercept:", lin_reg.intercept_)
    print("Scikit-learn coefficients:", lin_reg.coef_)
    # Note: Our theta = [intercept, coef1, coef2, coef3]
    # Compare theta[0] with lin_reg.intercept_ and the rest with lin_reg.coef_

if __name__ == "__main__":
    main()