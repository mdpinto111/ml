import pandas as pd


def make_onehot(df):
    onehot_hardcoded = []

    onehot_from_data = [col for col in df.columns if col.endswith("_onehot")]

    onehot_columns = onehot_from_data + onehot_hardcoded

    return pd.get_dummies(df, columns=onehot_columns, prefix_sep="_")
