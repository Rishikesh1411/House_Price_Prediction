"""
Utility Functions
Author: Rishikesh Raj
"""

import joblib

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)