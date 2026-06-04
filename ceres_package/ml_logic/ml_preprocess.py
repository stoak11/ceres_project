import pandas as pd

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
