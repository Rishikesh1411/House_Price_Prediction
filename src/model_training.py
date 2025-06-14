"""
Model Training Module
Author: Rishikesh Raj
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_decision_tree(X_train, y_train):
    param_grid = {
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "splitter": ["best", "random"],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    tree = DecisionTreeRegressor(random_state=42)
    grid = GridSearchCV(tree, param_grid, cv=3)
    grid.fit(X_train, y_train)
    return grid

def train_random_forest(X_train, y_train):
    param_grid = {
        "max_depth": [5, 10, 15],
        "n_estimators": [2, 5, 10],
    }
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3)
    grid.fit(X_train, y_train)
    return grid

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr