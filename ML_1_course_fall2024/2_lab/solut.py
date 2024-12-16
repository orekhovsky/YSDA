import numpy as np

def gradient_descent_linear_regression(
    target_values: np.ndarray,
    feature_matrix: np.ndarray,
    learning_rate: float = 0.01,
    num_iterations: int = 1000,
    tolerance: float = 1e-6
) -> np.ndarray:
    W =np.zeros(np.shape(feature_matrix[1]))
    for step in range(num_iterations):     
        f = np.dot(feature_matrix,W)
        err = f - target_values
        grad = (2*np.dot(feature_matrix.T,err))/len(target_values)
        W -= learning_rate*grad
        if np.linalg.norm(learning_rate*grad)<tolerance:
            break
    return W