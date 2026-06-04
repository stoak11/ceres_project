from pathlib import Path
from typing import Literal

from google.cloud import storage

import pandas as pd
import re

from ceres_package.params import *
from ceres_package.utils import simple_time_and_memory_tracker


from typing import Literal
from pathlib import Path


def load_from_gcp(source: DATA_SOURCE, dept: str | None = None) -> pd.DataFrame:
    if source not in DATA_CONFIG:
        raise ValueError(f"Source inconnue : '{source}'. Valeurs possibles : {list(DATA_CONFIG)}")

    config = DATA_CONFIG[source].copy()

    if source == 'meteo_dept':
        if dept is None:
            raise ValueError("L'argument 'dept' est requis pour la source 'meteo_dept'")
        config['blob'] = config['blob'].format(dept=dept)
        config['local'] = Path(config['local'].format(dept=dept))

    if source == 'meteo_dept':
        download_blob(source_blob_name=config['blob'], destination_file_name=config['local'])
        df = clean_meteo_data(config['local'])
    else :
        download_blob(source_blob_name=config['blob'], destination_file_name=config['local'])
        read_options = config.get('read_options', {}).copy()
        chunksize = read_options.pop('chunksize', None)
        dtype = read_options.pop('dtype', None)

        if chunksize:
            agg_config = config.get('agg_config', None)
            chunks = []
            for chunk in pd.read_csv(config['local'], chunksize=chunksize, **read_options):
                chunk = chunk[pd.to_numeric(chunk['altitude_m'], errors='coerce').notna()]
                if dtype:
                    chunk = chunk.astype({col: t for col, t in dtype.items() if col in chunk.columns})
                chunk['datetime'] = pd.to_datetime(chunk['datetime'])
                chunk['day'] = chunk['datetime'].dt.to_period('D')

                if agg_config:
                    agg = chunk.groupby(['dept_id', 'day']).agg(
                        **{col: (col, 'mean') for col in agg_config.get('mean', []) if col in chunk.columns},
                        **{col: (col, 'sum')  for col in agg_config.get('sum',  []) if col in chunk.columns})
                else:
                    agg = chunk.groupby(['dept_id', 'day']).mean(numeric_only=True)

                chunks.append(agg)

            df = pd.concat(chunks).groupby(['dept_id', 'day']).agg(
                **{col: (col, 'mean') for col in agg_config.get('mean', []) if col in chunks[0].columns},
                **{col: (col, 'sum')  for col in agg_config.get('sum',  []) if col in chunks[0].columns},
            ) if agg_config else pd.concat(chunks).groupby(['dept_id', 'day']).mean()
            df = df.reset_index()
        else:
            df = pd.read_csv(config['local'], **read_options)
            if dtype:
                df = df.astype({col: t for col, t in dtype.items() if col in df.columns})
            if 'day' in df.columns:
                df['day'] = pd.to_datetime(df['day'].astype(str))

    return df

@simple_time_and_memory_tracker
def clean_meteo_data(file_path) -> pd.DataFrame:
    dept_id = re.search(r'dept_(\w+)\.csv', file_path).group(1)
    chunks_agg = []
    """ Loop over one CSV """
    for chunk in pd.read_csv(file_path, chunksize=50_000, sep=',', encoding='utf-8-sig'):

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
def consolidate_meteo_data():
    """ Consolidate all departments CSV into one file """
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    first_write = True
    consolidation_file = os.path.join(BASE_DIR, 'raw_data', 'meteofrance', 'france', 'meteofrance_full.csv')

    for dept in DEPARTEMENTS_ID:
        file_path = os.path.join(BASE_DIR, 'raw_data', 'meteofrance', 'departements', f'dept_{dept}.csv')
        df_temp = clean_meteo_data(file_path=file_path)
        df_temp.to_csv(consolidation_file, mode='a', index=False, header=first_write)
        first_write = False
        print(f'✅ Département {dept} consolidated to full file')
        del df_temp

    print(f'\n✅ Terminé — {consolidation_file}')

def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS object
    # source_blob_name = récupérer le chemin du fichier dans le bucket, sans le nom du bucket (ex:'SAA-prod-ble/RAW_Data/SAA_2010-2025_provisoires_donnees_departementales.xlsx')

    # The path to which the file should be downloaded
    # destination_file_name = une liste contenant le chemin du fichier (ex: ['raw_data', 'agrestesaa','SAA_2010-2025_provisoires_donnees_departementales.xlsx'])

    storage_client = storage.Client(project=GCP_PROJECT)

    bucket = storage_client.bucket(BUCKET_NAME)

    destination_file_name_path = Path(destination_file_name)


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
                source_blob_name, BUCKET_NAME, destination_file_name)
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
    local_path = ROOT / 'raw_data' / 'agrestesaa' / 'SAA_2010-2025_provisoires_donnees_departementales.xlsx'

    download_blob(source_blob_name='SAA-prod-ble/RAW_Data/SAA_2010-2025_provisoires_donnees_departementales.xlsx', destination_file_name=local_path)

    df = pd.read_excel(local_path, sheet_name='COP')

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

    df = df.rename(columns={"LIB_DEP": "DEPT"})
    df = df.rename(columns={"LIB_SAA": "TYPE BLE"})

    # ------------------------------------------------------------------
    # Passage du format large au format long
    # ------------------------------------------------------------------

    # Rendement
    df_rend = pd.melt(
        df,
        id_vars=["DEPT_ID", "DEPT", "TYPE BLE", "STATUT_QUALITE"],
        value_vars=[col for col in df.columns if col.startswith("REND_")],
        var_name="ANNEE",
        value_name="RENDEMENT",
    )

    df_rend["ANNEE"] = df_rend["ANNEE"].str.removeprefix("REND_")

    # Surface
    df_surf = pd.melt(
        df,
        id_vars=["DEPT_ID", "DEPT", "TYPE BLE", "STATUT_QUALITE"],
        value_vars=[col for col in df.columns if col.startswith("SURF_")],
        var_name="ANNEE",
        value_name="SURFACE",
    )

    df_surf["ANNEE"] = df_surf["ANNEE"].str.removeprefix("SURF_")

    # Production
    df_prod = pd.melt(
        df,
        id_vars=["DEPT_ID", "DEPT", "TYPE BLE", "STATUT_QUALITE"],
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
            "DEPT",
            "TYPE BLE",
            "STATUT_QUALITE",
            "ANNEE",
        ],
    )

    target_ble_hiver = target_ble_hiver.merge(
        df_rend,
        on=[
            "DEPT_ID",
            "DEPT",
            "TYPE BLE",
            "STATUT_QUALITE",
            "ANNEE",
        ],
    )

    return target_ble_hiver


if __name__ == '__main__':
    clean_production_data()
