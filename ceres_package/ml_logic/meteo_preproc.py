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

########### PREPROC DL ###########
def add_datetime_features_dl(df: pd.DataFrame) -> pd.DataFrame:
    # Conversion to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Vectorial Extraction
    dt = df['datetime'].dt
    df['heure'] = dt.hour.astype('int8')
    df['jour']  = dt.day.astype('int8')
    df['mois']  = dt.month.astype('int8')
    df['annee'] = dt.year.astype('int16')

    # Saison agronomique — map sur les valeurs uniques pour éviter 13M appels
    unique_dates = df['datetime'].drop_duplicates()
    season_map = {ts: get_crop_season(ts) for ts in unique_dates}
    df['saison'] = df['datetime'].map(season_map).astype('category')

    # Encoding dept_id en 2 digits (01, 02, ..., 95)
    df['DEPT_ID'] = df['dept_id'].astype(int).astype(str).str.zfill(2)
    df = df.drop(columns='dept_id')

    return df

def add_harvest_year_dl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["harvest_year"] = df["datetime"].dt.year + (df["datetime"].dt.month >= 9).astype(int)
    return df

def time_cycle_dl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables temporelles cycliques en sin/cos en place.
    """
    # --- Heure (0-23) ---
    df['heure_sin'] = np.sin(2 * np.pi * df['heure'] / 24).astype('float32')
    df['heure_cos'] = np.cos(2 * np.pi * df['heure'] / 24).astype('float32')

    # --- Jour (1-31) ---
    df['jour_sin'] = np.sin(2 * np.pi * df['jour'] / 31).astype('float32')
    df['jour_cos'] = np.cos(2 * np.pi * df['jour'] / 31).astype('float32')

    # --- Mois (1-12) ---
    df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12).astype('float32')
    df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12).astype('float32')

    # --- Saison agronomique (6 stades) ---
    saison_order = {
        'semis': 0, 'vernalisation': 1, 'tallage': 2,
        'montaison': 3, 'floraison': 4, 'remplissage': 5,
    }
    saison_num = df['saison'].map(saison_order)
    df['saison_sin'] = np.sin(2 * np.pi * saison_num / 6).astype('float32')
    df['saison_cos'] = np.cos(2 * np.pi * saison_num / 6).astype('float32')

    # Supprimer les colonnes originales en place
    df.drop(columns=['heure', 'jour', 'mois', 'saison'], inplace=True)

    return df


########### PREPROC ML ###########
