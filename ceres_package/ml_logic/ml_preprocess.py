import pandas as pd
from ceres_package.params import *
from ceres_package.ml_logic.meteo_preproc import add_datetime_features_ml, add_harvest_year_ml, fast_impute_ml, get_crop_season
from ceres_package.ml_logic.data import load_from_gcp

def create_clean_target(dftarget):
    dftarget_clean = dftarget.dropna(subset = ["RENDEMENT"])
    dftarget_clean = dftarget_clean.drop(columns=["TYPE BLE", "SURFACE", "PRODUCTION"])
    dftarget_clean = dftarget_clean.rename(columns={"REGION": "DEPT"})
    dftarget_clean = dftarget_clean[(dftarget_clean["DEPT_ID"] != "2A") & (dftarget_clean["DEPT_ID"] != "2B")]

    return dftarget_clean

def merge_sol_y(dftarget_clean, dsol):
    merged_df = pd.merge(dftarget_clean, dsol, on=["DEPT_ID"], how="inner")
    merged_df = merged_df.drop(columns=["dept_nom"])

    return merged_df

def merge_dataframes(df1, df2):
    merged_df = pd.merge(df1, df2, on=["DEPT_ID", "ANNEE"], how="inner")

    return merged_df

def merge_dataframes_meteo (df1, df2):
    merged_df = pd.merge(df1, df2, on=["DEPT_ID", "harvest_year"], how="inner")

    return merged_df



def preprocess_ndvi(df_ndvi):
    """
    Nettoie et agrège les données NDVI par département et année.

    - Renomme les colonnes (department_name → DEPT, year → ANNEE)
    - Calcule la moyenne du NDVI par DEPT_ID, DEPT et ANNEE

    Returns
    -------
    DataFrame agrégé avec une ligne par département et par année.
    """

    df_ndvi = df_ndvi.rename(columns={'department_name': 'DEPT', 'year':'ANNEE'})
    df_ndvi_preproc = (
        df_ndvi
        .groupby(['DEPT_ID', 'DEPT', 'ANNEE'])['ndvi_poly_mean']
        .mean()
        .reset_index()
        )

    return df_ndvi_preproc


def preprocess_soil(df_soil):
    """
    Standardise les données de sol au niveau départemental.

    - Renomme la colonne 'dept_nom' en 'DEPT' pour harmoniser les clés de jointure
      avec les autres jeux de données.

    Returns
    -------
    DataFrame contenant les variables de sol avec une clé département harmonisée.
    """

    df_soil_preproc = df_soil.rename(columns={'dept_nom':'DEPT'})

    return df_soil_preproc


######## METEO ML ###########

def preprocess_meteo_annee(df_meteo):
    """
    Agrège les données météorologiques quotidiennes à l échelle annuelle et départementale.

    La fonction :
    - Harmonise les codes département (DEPT_ID au format 2 chiffres)
    - Standardise les noms de colonnes
    - Extrait l année depuis la colonne 'day'
    - Agrège les variables météo selon les règles définies dans DATA_CONFIG
      (moyenne ou somme selon les variables)

    Returns
    -------
    pandas.DataFrame
        DataFrame agrégé avec une ligne par DEPT_ID et ANNEE.
    """

    # Harmonisation des codes département (01, 02, ..., 95)
    df_meteo["dept_id"] = df_meteo["dept_id"].astype(str).str.zfill(2)

    # Standardisation du nom de colonne
    df_meteo = df_meteo.rename(columns={"dept_id": "DEPT_ID"})

    # Extraction de l'année depuis la date
    df_meteo["ANNEE"] = df_meteo["day"].dt.year

    # Récupération des règles d'agrégation
    agg_config = DATA_CONFIG["meteo_full"]["agg_config"]
    meteo_mean = agg_config["mean"]
    meteo_sum = agg_config["sum"]

    # Construction du dictionnaire d'agrégation pandas
    agg_dict = {col: "mean" for col in meteo_mean}
    agg_dict.update({col: "sum" for col in meteo_sum})

    # Agrégation annuelle par département
    df_meteo_agg = (
        df_meteo
        .groupby(["DEPT_ID", "ANNEE"])
        .agg(agg_dict)
        .reset_index()
    )

    return df_meteo_agg

def preprocess_meteo_ml(df_meteo):

    df_meteo = add_harvest_year_ml(df_meteo)

        # datetime features UNE SEULE FOIS
    df_meteo = add_datetime_features_ml(df_meteo)

    # filtre
    df_meteo = df_meteo[~df_meteo["DEPT_ID"].isin([92, 93, 94, 75])]

    # drop features
    df_meteo = df_meteo.drop(columns=[
        'duree_humectation_foliaire_min',
        'temp_sol_100cm_c',
        'temp_sol_50cm_c',
        'temp_sol_20cm_c',
        'etat_sol',
        'pression_mer_hpa',
        'temp_surface_sol_c'
    ])

    # imputation
    cols_valeurs = ['temp_air_c', 'temp_min_c', 'temp_max_c', 'temp_rosee_c',
        'humidite_relative_pct', 'humidite_min_pct', 'humidite_max_pct',
        'vent_moyen_10m_ms', 'rafale_max_ms', 'tension_vapeur_hpa',
        'temp_sol_10cm_c', 'temp_min_sol_10cm_c', 'heure_humidite_min',
        'heure_humidite_max', 'direction_vent_deg', 'hauteur_neige_sol_cm',
        'precipitations_1h_mm', 'duree_precipitations_min', 'duree_gel_min',
        'insolation_min', 'rayonnement_global_jcm2', 'duree_humidite_inf40_min',
        'duree_humidite_sup80_min']

    df_meteo = fast_impute_ml(df_meteo, cols_valeurs)

    return df_meteo
