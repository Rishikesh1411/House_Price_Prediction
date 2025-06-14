"""
Model Evaluation Module
Author: Rishikesh Raj
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    return {"MAE": mae, "MSE": mse}