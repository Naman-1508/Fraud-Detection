
import pandas as pd

def preprocess(df):
    X = df.drop(columns=["Class", "Time"])
    X["Amount"] = (X["Amount"] - X["Amount"].mean()) / (X["Amount"].std() + 1e-9)
    y = df["Class"]
    return X, y
