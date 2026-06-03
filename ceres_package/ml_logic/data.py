from pathlib import Path

from google.cloud import storage
import pandas as pd
import matplotlib.pyplot as plt
import re

from ceres_package.params import *
from ceres_package.utils import simple_time_and_memory_tracker


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

@simple_time_and_memory_tracker
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
def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS object
    # source_blob_name = récupérer le chemin du fichier dans le bucket, sans le nom du bucket (ex:'SAA-prod-ble/RAW_Data/SAA_2010-2025_provisoires_donnees_departementales.xlsx')

    # The path to which the file should be downloaded
    # destination_file_name = une liste contenant le chemin du fichier (ex: ['raw_data', 'agrestesaa','SAA_2010-2025_provisoires_donnees_departementales.xlsx'])

    storage_client = storage.Client(project=GCP_PROJECT)

    bucket = storage_client.bucket(BUCKET_NAME)

    destination_file_name_path = Path(*destination_file_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    if destination_file_name_path.is_file() is False:
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name_path)

        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
                source_blob_name, BUCKET_NAME, '/'.join(destination_file_name))
            )
    else:
        print(
            "Storage object {} from bucket {} to local file {} already downloaded.".format(
                source_blob_name, BUCKET_NAME, '/'.join(destination_file_name))
            )

def clean_production_data():
    """
    Télécharge et nettoie les données de production agricole.

    Étapes :
    - Chargement de la feuille COP.
    - Nettoyage de la structure du fichier Excel.
    - Sélection du blé tendre d'hiver.
    - Suppression des départements d'outre-mer.
    - Création d'un identifiant département.
    - Restructuration des séries temporelles (surface, production, rendement).
    - Fusion des indicateurs dans une table finale.

    Returns
    -------
    pandas.DataFrame
        Table au format long contenant :
        - DEPT_ID
        - REGION
        - TYPE BLE
        - STATUT_QUALITE
        - ANNEE
        - SURFACE
        - PRODUCTION
        - RENDEMENT
    """

    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------
    local_path = ['raw_data', 'agrestesaa','SAA_2010-2025_provisoires_donnees_departementales.xlsx']

    blob = download_blob(source_blob_name='SAA-prod-ble/RAW_Data/SAA_2010-2025_provisoires_donnees_departementales.xlsx', destination_file_name=local_path)

    df = pd.read_excel(Path(*local_path), sheet_name='COP')

    # ------------------------------------------------------------------
    # Nettoyage de la structure du fichier Excel
    # ------------------------------------------------------------------

    # Suppression des lignes d'en-tête inutiles
    df = df.drop([0, 1, 2, 3])

    # Utilisation de la première ligne comme noms de colonnes
    df.columns = df.iloc[0]

    # Suppression de la ligne utilisée comme en-tête
    df = df.iloc[1:].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Filtrage de la culture étudiée
    # ------------------------------------------------------------------

    # Harmonisation des libellés
    df["LIB_SAA"] = df["LIB_SAA"].str.lower()

    # Conservation du blé tendre d'hiver uniquement
    bth = "blé tendre d'hiver"

    mask = df["LIB_SAA"].str.contains(bth)
    df = df[mask]

    # ------------------------------------------------------------------
    # Nettoyage géographique
    # ------------------------------------------------------------------

    # Exclusion des DOM-TOM
    mask_metropole = df["LIB_DEP"].str.startswith("0")
    df = df[mask_metropole]

    # Suppression du zéro initial des codes département
    df["LIB_DEP"] = df["LIB_DEP"].str.removeprefix("0")

    # Création de l'identifiant département
    df["DEPT_ID"] = df["LIB_DEP"].str[:2]

    col = df.pop("DEPT_ID")
    df.insert(0, "DEPT_ID", col)

    # Nettoyage et renommage des colonnes
    df = df.drop(columns="LIB_REG2")

    df["LIB_DEP"] = df["LIB_DEP"].str.slice(4)

    df = df.rename(columns={"LIB_DEP": "REGION"})
    df = df.rename(columns={"LIB_SAA": "TYPE BLE"})

    # ------------------------------------------------------------------
    # Passage du format large au format long
    # ------------------------------------------------------------------

    # Rendement
    df_rend = pd.melt(
        df,
        id_vars=["DEPT_ID", "REGION", "TYPE BLE", "STATUT_QUALITE"],
        value_vars=[col for col in df.columns if col.startswith("REND_")],
        var_name="ANNEE",
        value_name="RENDEMENT",
    )

    df_rend["ANNEE"] = df_rend["ANNEE"].str.removeprefix("REND_")

    # Surface
    df_surf = pd.melt(
        df,
        id_vars=["DEPT_ID", "REGION", "TYPE BLE", "STATUT_QUALITE"],
        value_vars=[col for col in df.columns if col.startswith("SURF_")],
        var_name="ANNEE",
        value_name="SURFACE",
    )

    df_surf["ANNEE"] = df_surf["ANNEE"].str.removeprefix("SURF_")

    # Production
    df_prod = pd.melt(
        df,
        id_vars=["DEPT_ID", "REGION", "TYPE BLE", "STATUT_QUALITE"],
        value_vars=[col for col in df.columns if col.startswith("PROD_")],
        var_name="ANNEE",
        value_name="PRODUCTION",
    )

    df_prod["ANNEE"] = df_prod["ANNEE"].str.removeprefix("PROD_")

    # ------------------------------------------------------------------
    # Fusion des indicateurs
    # ------------------------------------------------------------------

    target_ble_hiver = df_surf.merge(
        df_prod,
        on=[
            "DEPT_ID",
            "REGION",
            "TYPE BLE",
            "STATUT_QUALITE",
            "ANNEE",
        ],
    )

    target_ble_hiver = target_ble_hiver.merge(
        df_rend,
        on=[
            "DEPT_ID",
            "REGION",
            "TYPE BLE",
            "STATUT_QUALITE",
            "ANNEE",
        ],
    )

    return target_ble_hiver

def clean_meteo_data(df : pd.DataFrame) -> pd.DataFrame :
    pass

if __name__ == '__main__':
    clean_production_data()
