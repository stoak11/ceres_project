import pandas as pd

from ceres_package.params import *

def get_crop_season(date: pd.Timestamp) -> str:
    month, day = date.month, date.day

    if month in (9, 10, 11):
        return 'semis'
    elif month == 12 or month in (1, 2):
        return 'vernalisation'
    elif month == 3 or (month == 4 and day <= 15):
        return 'tallage'
    elif (month == 4 and day > 15) or (month == 5 and day <= 15):
        return 'montaison'
    elif (month == 5 and day > 15) or (month == 6 and day <= 15):
        return 'floraison'
    elif (month == 6 and day > 15) or month in (7, 8):
        return 'remplissage'
    raise ValueError(f"Date inattendue : {date}")




########### PREPROC ML ###########

def add_datetime_features_ml(df: pd.DataFrame) -> pd.DataFrame:
    # Conversion to datetime
    df['day'] = pd.to_datetime(df['day'])

    # Vectorial Extraction
    dt = df['day'].dt
    df['heure'] = dt.hour.astype('int8')
    df['jour']  = dt.day.astype('int8')
    df['mois']  = dt.month.astype('int8')
    df['annee'] = dt.year.astype('int16')

    # Saison agronomique — map sur les valeurs uniques pour éviter 13M appels
    unique_dates = df['day'].drop_duplicates()
    season_map = {ts: get_crop_season(ts) for ts in unique_dates}
    df['saison'] = df['day'].map(season_map).astype('category')

    # Encoding dept_id en 2 digits (01, 02, ..., 95)
    df['DEPT_ID'] = df['dept_id'].astype(int).astype(str).str.zfill(2)
    df = df.drop(columns='dept_id')

    return df

def add_harvest_year_ml(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["harvest_year"] = df["day"].dt.year + (df["day"].dt.month >= 9).astype(int)
    return df

def fast_impute_ml(df, cols):

    df = df.copy()

    for col in cols:

        # 1. médiane par département + année
        df[col] = df[col].fillna(
            df.groupby(['DEPT_ID', 'harvest_year'])[col].transform('median')
        )

        # 2. médiane par département
        df[col] = df[col].fillna(
            df.groupby('DEPT_ID')[col].transform('median')
        )

    return df
