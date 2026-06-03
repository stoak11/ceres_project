import pandas as pd
from ceres_package.utils import simple_time_and_memory_tracker
from ceres_package.params import *
import re

@simple_time_and_memory_tracker


def clean_meteo_data(gcp_path) -> pd.DataFrame :
    """ Import CSV locally """
    # gcp_path = '/home/berlh/code/stoak11/ceres_project/raw_data/meteofrance/dept_59.csv'
    df = pd.read_csv(gcp_path, sep=',', encoding='utf-8-sig')

    """ Drop useless columns """
    cols_to_drop = [c for c in df.columns if c not in COLONNES_BLE]
    df_lite = df.drop(columns=cols_to_drop)
    print(f"Colonnes supprimées : {len(cols_to_drop)}")
    print(f"Colonnes restantes  : {df_lite.shape[1]}")

    """ Convert Météo France timestamps """
    df_lite["DATE"] = pd.to_datetime(df_lite["AAAAMMJJHH"].astype(str), format="%Y%m%d%H")
    df_timeconvert = df_lite.drop(columns=["AAAAMMJJHH"])

    print('🕘✅ Time Stamp Converted')

    """ Rename columns with self-explanatory names """
    df_col_renamed = df_timeconvert.rename(columns=RENAME_COLONNES_BLE)

    """ Colonnes de mesure (hors identifiants station) """
    mesure_cols = [c for c in df_col_renamed.columns if c not in ["id_station", "latitude", "longitude", "datetime", "dept_id"]]

    """" Aggregation by timestamp """
    df_agg = df_col_renamed.groupby("datetime")[mesure_cols].mean().reset_index()

    print(f"Shape avant agrégation : {df_col_renamed.shape}")
    print(f"Shape après agrégation : {df_agg.shape}")

    """ Add column ['dept_id'] """
    dept_id = re.search(r'dept_(\w+)\.csv', gcp_path).group(1)
    df_agg["dept_id"] = dept_id

    return df_agg
