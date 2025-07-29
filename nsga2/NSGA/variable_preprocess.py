from .problem_variables import ProblemVariables
import numpy as np


def variable_scaling(X, vars_obj):
    # Ensure X is a NumPy array of type float
    X = np.asarray(X, dtype=np.float64)
    
    # Check the dimensions compatibility.
    if X.shape[1] != vars_obj.get_num_of_variables():
        raise ValueError("Dimension mismatch: X columns do not match number of variables.")
    
    scale_values = vars_obj.get_scale_values()

    # Ensure scaling for each variable.
    for i, scale in enumerate(scale_values):
        X[:, i] = X[:, i] / scale  # divide each column by corresponding scale value
        
    return X


def extract_variables(X, vars_obj):
    var_names = vars_obj.get_names()
    variables = {name: X[:, i] for i, name in enumerate(var_names)}
    return variables