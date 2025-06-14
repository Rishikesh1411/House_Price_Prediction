"""
Data Preprocessing Module
Author: Rishikesh Raj
"""

import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def get_features_and_target(df):
    X = df[["number of bedrooms", "number of bathrooms", "living area", "condition of the house", "Number of schools nearby"]]
    y = df["Price"]
    return X, y