import pandas as pd
from ceres_package.utils import simple_time_and_memory_tracker
from ceres_package.params import *
import re

@simple_time_and_memory_tracker


def clean_meteo_data(gcp_path) -> pd.DataFrame:

    dept_id = re.search(r'dept_(\w+)\.csv', gcp_path).group(1)
    chunks_agg = []
    """ Loop over one CSV """
    for chunk in pd.read_csv(gcp_path, chunksize=50_000, sep=',', encoding='utf-8-sig'):

        """ Drop useless columns """
        cols_to_drop = [c for c in chunk.columns if c not in COLONNES_BLE]
        chunk = chunk.drop(columns=cols_to_drop)

        """ Convert Météo France timestamps """
        chunk["DATE"] = pd.to_datetime(chunk["AAAAMMJJHH"].astype(str), format="%Y%m%d%H")
        chunk = chunk.drop(columns=["AAAAMMJJHH"])

        """ Rename columns with self-explanatory names """
        chunk = chunk.rename(columns=RENAME_COLONNES_BLE)
        """ Add cleaned chunks to chunks list and delete current chunk from RAM """
        chunks_agg.append(chunk)
        del chunk

    """ Concat cleaned chunks list and delete current chunks list from RAM """
    df_dept = pd.concat(chunks_agg, ignore_index=True)
    del chunks_agg

    """ Colonnes de mesure (hors identifiants station) """
    mesure_cols = [c for c in df_dept.columns if c not in ["id_station", "latitude", "longitude", "datetime"]]

    """" Aggregation by timestamp and delete pre-aggregation DataFrame from RAM """
    df_agg = df_dept.groupby("datetime")[mesure_cols].mean(numeric_only=True).reset_index()
    del df_dept

    """ Add column ['dept_id'] """
    df_agg["dept_id"] = dept_id

    print(f'✅ Département {dept_id} cleaned — {len(df_agg):,} lignes')
    return df_agg


def consolidate_meteo_data(gcp_path, consolidated_file):
    """ In Case of future update to consolidated all updated departments CSV """
    first_write = True

    for dept in range(21, 96):
        dept_str = f'{dept:02d}'
        gcp_path = f'/home/berlh/code/stoak11/ceres_project/raw_data/meteofrance/departements/dept_{dept_str}.csv'
        df_temp = clean_meteo_data(gcp_path=gcp_path)
        df_temp.to_csv(consolidated_file, mode='a', index=False, header = first_write)
        first_write = False
        print(f'✅ Département {dept_str} consolidated to full file')
        del df_temp

    print(f'\n✅ Terminé — {consolidated_file}')
