import pandas as pd

def create_dummy_vars(df):


    # Separate the input features and target variable
    x = df.drop('price', axis=1)
    y = df['price']

    return x, y