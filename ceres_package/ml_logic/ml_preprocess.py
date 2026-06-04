import pandas as pd

def create_clean_target(dftarget):
    dftarget_clean = dftarget.dropna(subset = ["RENDEMENT"])
    dftarget_clean = dftarget_clean.drop(columns=["REGION", "TYPE BLE", "SURFACE", "PRODUCTION"])

    return dftarget_clean

def merge_dataframes(df1, df2):
    merged_df = pd.merge(df1, df2, on="DEPT_ID", how="inner")
    merged_df = merged_df.drop(columns=["dept_nom"])

    return merged_df
