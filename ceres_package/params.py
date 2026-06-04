import os
import numpy as np
from typing import Literal
from pathlib import Path

##################  VARIABLES  ##################
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")



##################  CONSTANTS  #####################
# LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
# LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")


##################  SOURCE  #####################
ROOT = Path(__file__).resolve().parents[1]


DATA_SOURCE = Literal[
    'production',
    'meteo_full',
    'meteo_dept',
    'soil',
    'ndvi_season',
    'ndvi_month',
    ]

METEO_DTYPES = {
    'altitude_m': 'float32',
    'precipitations_1h_mm': 'float32',
    'duree_precipitations_min': 'float32',
    'vent_moyen_10m_ms': 'float32',
    'direction_vent_deg': 'float32',
    'rafale_max_ms': 'float32',
    'temp_air_c': 'float32',
    'temp_rosee_c': 'float32',
    'temp_min_c': 'float32',
    'temp_max_c': 'float32',
    'duree_gel_min': 'float32',
    'temp_sol_10cm_c': 'float32',
    'temp_sol_20cm_c': 'float32',
    'temp_sol_50cm_c': 'float32',
    'temp_sol_100cm_c': 'float32',
    'temp_min_sol_10cm_c': 'float32',
    'temp_surface_sol_c': 'float32',
    'duree_humectation_foliaire_min': 'float32',
    'humidite_relative_pct': 'float32',
    'humidite_min_pct': 'float32',
    'heure_humidite_min': 'float32',
    'humidite_max_pct': 'float32',
    'heure_humidite_max': 'float32',
    'duree_humidite_inf40_min': 'float32',
    'duree_humidite_sup80_min': 'float32',
    'tension_vapeur_hpa': 'float32',
    'pression_mer_hpa': 'float32',
    'etat_sol': 'float32',
    'hauteur_neige_sol_cm': 'float32',
    'rayonnement_global_jcm2': 'float32',
    'insolation_min': 'float32',
    'temp_min_sol_10cm_c': 'float32',
    'heure_humidite_min': 'float32',
    'heure_humidite_max': 'float32',
    'direction_vent_deg': 'float32',
    'hauteur_neige_sol_cm': 'float32',
    'etat_sol': 'float32',
    'duree_humectation_foliaire_min': 'float32',
    'dept_id': 'int32',
}

DATA_CONFIG: dict[str, dict] = {
    'production': {
        'blob': 'SAA-prod-ble/Clean_Data/target_ble_tendre_hiver.csv',
        'local': ROOT / 'raw_data' / 'agrestesaa' / 'clean_wheat_prod.csv',
        },
    'meteo_full': {
        'blob': 'meteo_france_data/france/meteofrance_full.csv',
        'local': ROOT / 'raw_data' / 'meteofrance' / 'france' / 'meteofrance_full.csv',
        'read_options': {
        'dtype': METEO_DTYPES,
        'chunksize': 50_000,
        'low_memory': False},
        'agg_config': {
        'mean': [
            'temp_air_c', 'temp_min_c', 'temp_max_c', 'temp_rosee_c',
            'humidite_relative_pct', 'humidite_min_pct', 'humidite_max_pct',
            'vent_moyen_10m_ms', 'rafale_max_ms', 'tension_vapeur_hpa',
            'pression_mer_hpa', 'temp_sol_10cm_c', 'temp_sol_20cm_c',
            'temp_sol_50cm_c', 'temp_sol_100cm_c', 'temp_surface_sol_c',
            'temp_min_sol_10cm_c', 'heure_humidite_min', 'heure_humidite_max',
            'direction_vent_deg', 'hauteur_neige_sol_cm', 'etat_sol',
            'duree_humectation_foliaire_min'
            ],
        'sum': [
            'precipitations_1h_mm', 'duree_precipitations_min',
            'duree_gel_min', 'insolation_min', 'rayonnement_global_jcm2',
            'duree_humidite_inf40_min', 'duree_humidite_sup80_min'
            ],
            }
        },
    'meteo_dept': {
        'blob': 'meteo_france_data/departements/dept_{dept}.csv',
        'local': str(ROOT / 'raw_data' / 'meteofrance' / 'departements' / 'dept_{dept}.csv'),
        },
    'soil': {
        'blob': 'propriétés de sol/soilgrids.csv',
        'local': ROOT / 'raw_data' / 'soil_grid' / 'soil_grids.csv',
        },
    'ndvi_season': {
        'blob': 'NDVI/ndvi_season_features.csv',
        'local': ROOT / 'raw_data' / 'ndvi' / 'ndvi_season.csv',
        },
    'ndvi_month': {
        'blob': 'NDVI/ndvi_monthly_by_department_polygon.csv',
        'local': ROOT / 'raw_data' / 'ndvi' / 'ndvi_monthly.csv',
        },
    }
#################  Departements  ####################
DEPARTEMENTS = {
            '01': 'Ain',
            '02': 'Aisne',
            '03': 'Allier',
            '04': 'Alpes-de-Haute-Provence',
            '05': 'Hautes-Alpes',
            '06': 'Alpes-Maritimes',
            '07': 'Ardèche',
            '08': 'Ardennes',
            '09': 'Ariège',
            '10': 'Aube',
            '11': 'Aude',
            '12': 'Aveyron',
            '13': 'Bouches-du-Rhône',
            '14': 'Calvados',
            '15': 'Cantal',
            '16': 'Charente',
            '17': 'Charente-Maritime',
            '18': 'Cher',
            '19': 'Corrèze',
            '21': "Côte-d'Or",
            '22': "Côtes-d'Armor",
            '23': 'Creuse',
            '24': 'Dordogne',
            '25': 'Doubs',
            '26': 'Drôme',
            '27': 'Eure',
            '28': 'Eure-et-Loir',
            '29': 'Finistère',
            '2A': 'Corse-du-Sud',
            '2B': 'Haute-Corse',
            '30': 'Gard',
            '31': 'Haute-Garonne',
            '32': 'Gers',
            '33': 'Gironde',
            '34': 'Hérault',
            '35': 'Ille-et-Vilaine',
            '36': 'Indre',
            '37': 'Indre-et-Loire',
            '38': 'Isère',
            '39': 'Jura',
            '40': 'Landes',
            '41': 'Loir-et-Cher',
            '42': 'Loire',
            '43': 'Haute-Loire',
            '44': 'Loire-Atlantique',
            '45': 'Loiret',
            '46': 'Lot',
            '47': 'Lot-et-Garonne',
            '48': 'Lozère',
            '49': 'Maine-et-Loire',
            '50': 'Manche',
            '51': 'Marne',
            '52': 'Haute-Marne',
            '53': 'Mayenne',
            '54': 'Meurthe-et-Moselle',
            '55': 'Meuse',
            '56': 'Morbihan',
            '57': 'Moselle',
            '58': 'Nièvre',
            '59': 'Nord',
            '60': 'Oise',
            '61': 'Orne',
            '62': 'Pas-de-Calais',
            '63': 'Puy-de-Dôme',
            '64': 'Pyrénées-Atlantiques',
            '65': 'Hautes-Pyrénées',
            '66': 'Pyrénées-Orientales',
            '67': 'Bas-Rhin',
            '68': 'Haut-Rhin',
            '69': 'Rhône',
            '70': 'Haute-Saône',
            '71': 'Saône-et-Loire',
            '72': 'Sarthe',
            '73': 'Savoie',
            '74': 'Haute-Savoie',
            '76': 'Seine-Maritime',
            '77': 'Seine-et-Marne',
            '78': 'Yvelines',
            '79': 'Deux-Sèvres',
            '80': 'Somme',
            '81': 'Tarn',
            '82': 'Tarn-et-Garonne',
            '83': 'Var',
            '84': 'Vaucluse',
            '85': 'Vendée',
            '86': 'Vienne',
            '87': 'Haute-Vienne',
            '88': 'Vosges',
            '89': 'Yonne',
            '90': 'Territoire de Belfort',
            '91': 'Essonne',
            '92': 'Hauts-de-Seine',
            '93': 'Seine-Saint-Denis',
            '94': 'Val-de-Marne',
            '95': "Val-d'Oise",
            }
DEPARTEMENTS_ID = [
    "01","02","03","04","05","06","07","08","09","10",
    "11","12","13","14","15","16","17","18","19","21",
    "22","23","24","25","26","27","28","29",
    "30","31","32","33","34","35","36","37","38","39",
    "40","41","42","43","44","45","46","47","48","49",
    "50","51","52","53","54","55","56","57","58","59",
    "60","61","62","63","64","65","66","67","68","69",
    "70","71","72","73","74","75","76","77","78","79",
    "80","81","82","83","84","85","86","87","88","89",
    "90","91","92","93","94","95"]

##############  DATA METEO FRANCE  ##################

COLONNES_BLE = [
        # Identifiants & localisation
        "NUM_POSTE","LAT", "LON", "ALTI", "AAAAMMJJHH",
        # Température
        "T", "TN", "TX", "TD", "TNSOL", "TCHAUSSEE", "T10", "T20", "T50", "T100",
        "DG",
        # Humidité
        "U", "UN", "HUN", "UX", "HUX", "DHUMI80", "DHUMI40", "DHUMEC", "TSV",
        # Précipitations
        "RR1", "DRR1",
        # Vent
        "FF", "FXI", "DD",
        # Rayonnement & ensoleillement
        "GLO", "INS",
        # Neige & sol
        "NEIGETOT", "SOL",
        # Pression
        "PMER"
        ]

RENAME_COLONNES_BLE = {
        # Identifiants & localisation
        "NUM_POSTE":  "id_station",
        "LAT":        "latitude",
        "LON":        "longitude",
        "ALTI":       "altitude_m",
        "DATE": "datetime",
        # Température
        "T":          "temp_air_c",
        "TN":         "temp_min_c",
        "TX":         "temp_max_c",
        "TD":         "temp_rosee_c",
        "TNSOL":      "temp_min_sol_10cm_c",
        "TCHAUSSEE":  "temp_surface_sol_c",
        "T10":        "temp_sol_10cm_c",
        "T20":        "temp_sol_20cm_c",
        "T50":        "temp_sol_50cm_c",
        "T100":       "temp_sol_100cm_c",
        "DG":         "duree_gel_min",
        # Humidité
        "U":          "humidite_relative_pct",
        "UN":         "humidite_min_pct",
        "HUN":        "heure_humidite_min",
        "UX":         "humidite_max_pct",
        "HUX":        "heure_humidite_max",
        "DHUMI80":    "duree_humidite_sup80_min",
        "DHUMI40":    "duree_humidite_inf40_min",
        "DHUMEC":     "duree_humectation_foliaire_min",
        "TSV":        "tension_vapeur_hpa",
        # Précipitations
        "RR1":        "precipitations_1h_mm",
        "DRR1":       "duree_precipitations_min",
        # Vent
        "FF":         "vent_moyen_10m_ms",
        "FXI":        "rafale_max_ms",
        "DD":         "direction_vent_deg",
        # Rayonnement
        "GLO":        "rayonnement_global_jcm2",
        "INS":        "insolation_min",
        # Neige & sol
        "NEIGETOT":   "hauteur_neige_sol_cm",
        "SOL":        "etat_sol",
        # Pression
        "PMER":       "pression_mer_hpa",
    }

###############  DATA SOIL GRID  ####################


#################  DATA NVDI  #######################
